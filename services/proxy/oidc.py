"""Google OIDC sign-in for the admin page: authorization code with PKCE (S256), ``state`` and ``nonce``.

- :func:`new_login` makes the per-attempt secrets; :func:`sign_login` / :func:`read_login` keep them in the
  short-lived ``__Host-decision_oauth`` cookie, signed with HMAC-SHA256 under the session key.
- :class:`GoogleOIDC` builds the authorization URL, exchanges the code (with the PKCE verifier) at Google's token
  endpoint and verifies the ID token: RS256 signature against Google's JWKS (cached, refreshed on an unknown
  ``kid`` at most once a minute), issuer, audience/``azp``, ``exp``/``iat`` with a small leeway, and ``nonce``.
  ``email_verified`` is returned to the caller, which decides (and audits) the sign-in.

Every failure raises :class:`OidcError`; its message is for logs only and never contains tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})
SCOPE = "openid email profile"
ALGORITHM = "RS256"

LOGIN_COOKIE = "__Host-decision_oauth"
LOGIN_TTL_S = 600
LEEWAY_S = 60
JWKS_TTL_S = 3600
JWKS_MIN_REFRESH_S = 60
DEFAULT_NEXT = "/admin"
MAX_NEXT = 512
NEXT_RE = re.compile(r"^/admin([/?#][^\\]*)?$")
_LOGIN_CONTEXT = b"decision-api oauth login v1\x00"


class OidcError(Exception):
    """Sign-in could not be completed (bad code, unreachable Google, invalid ID token)."""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ---------------------------------------------------------------- per-attempt secrets


def pkce_challenge(verifier: str) -> str:
    return _b64(hashlib.sha256(verifier.encode()).digest())


def pkce_pair() -> tuple[str, str]:
    """A PKCE ``(code_verifier, code_challenge)``: 43 base64url characters and their S256 challenge."""
    verifier = secrets.token_urlsafe(32)
    return verifier, pkce_challenge(verifier)


def safe_next(value: Any) -> str:
    """``value`` when it is a same-origin path under ``/admin``, else ``/admin``."""
    if (
        isinstance(value, str)
        and len(value) <= MAX_NEXT
        and NEXT_RE.fullmatch(value)
        and all(0x21 <= ord(c) < 0x7F for c in value)
    ):
        return value
    return DEFAULT_NEXT


@dataclass(frozen=True)
class LoginState:
    state: str
    nonce: str
    verifier: str
    next: str


def new_login(next_: Any) -> LoginState:
    return LoginState(
        state=secrets.token_urlsafe(32),
        nonce=secrets.token_urlsafe(32),
        verifier=pkce_pair()[0],
        next=safe_next(next_),
    )


def _login_mac(key: bytes, payload: str) -> str:
    return _b64(hmac.new(key, _LOGIN_CONTEXT + payload.encode(), hashlib.sha256).digest())


def sign_login(key: bytes, login: LoginState, now: float) -> str:
    """The login cookie value: base64url JSON payload (with its expiry) and its HMAC."""
    body = {"s": login.state, "n": login.nonce, "v": login.verifier, "x": login.next, "e": int(now) + LOGIN_TTL_S}
    payload = _b64(json.dumps(body, separators=(",", ":")).encode())
    return f"{payload}.{_login_mac(key, payload)}"


def read_login(key: bytes, value: str | None, now: float) -> LoginState | None:
    """The login state from a cookie value, or None when it is missing, forged, malformed or expired."""
    if not value or value.count(".") != 1:
        return None
    payload, mac = value.split(".")
    if not hmac.compare_digest(mac.encode(), _login_mac(key, payload).encode()):
        return None
    try:
        body = json.loads(_unb64(payload))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    fields = [body.get(k) for k in ("s", "n", "v", "x")]
    expires = body.get("e")
    if not all(isinstance(f, str) and f for f in fields) or not isinstance(expires, int) or expires <= now:
        return None
    return LoginState(state=fields[0], nonce=fields[1], verifier=fields[2], next=safe_next(fields[3]))


# ---------------------------------------------------------------- Google


class GoogleOIDC:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        now: Callable[[], float] = time.time,
    ):
        self.client_id = client_id
        self._client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.transport = transport
        self.timeout = httpx.Timeout(10.0, connect=5.0)
        self._now = now
        self._keys: dict[str, Any] = {}
        self._jwks_at: float | None = None

    def __repr__(self) -> str:  # never show the client secret
        return f"GoogleOIDC(client_id={self.client_id!r}, redirect_uri={self.redirect_uri!r})"

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, trust_env=False, follow_redirects=False
        )

    def authorization_url(self, login: LoginState) -> str:
        query = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": SCOPE,
            "state": login.state,
            "nonce": login.nonce,
            "code_challenge": pkce_challenge(login.verifier),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        return f"{AUTH_URL}?{urlencode(query)}"

    async def exchange(self, code: str, verifier: str) -> str:
        """Trade the authorization code (and PKCE verifier) for the raw ID token."""
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "client_id": self.client_id,
            "client_secret": self._client_secret,
            "redirect_uri": self.redirect_uri,
        }
        try:
            async with self._http() as http:
                resp = await http.post(TOKEN_URL, data=form, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise OidcError(f"token endpoint unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise OidcError(f"token endpoint answered HTTP {resp.status_code}")
        try:
            token = resp.json().get("id_token")
        except (ValueError, AttributeError):
            token = None
        if not isinstance(token, str) or not token:
            raise OidcError("token endpoint returned no id_token")
        return token

    async def _refresh_jwks(self) -> None:
        try:
            async with self._http() as http:
                resp = await http.get(JWKS_URL)
            resp.raise_for_status()
            keys = resp.json()["keys"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise OidcError(f"JWKS fetch failed ({type(exc).__name__})") from None
        parsed = {}
        for jwk in keys if isinstance(keys, list) else []:
            if not isinstance(jwk, dict) or jwk.get("kty") != "RSA" or not isinstance(jwk.get("kid"), str):
                continue
            try:
                parsed[jwk["kid"]] = jwt.PyJWK(jwk, algorithm=ALGORITHM).key
            except jwt.PyJWTError:
                continue
        self._keys = parsed
        self._jwks_at = self._now()

    async def _signing_key(self, kid: str):
        now = self._now()
        stale = self._jwks_at is None or now - self._jwks_at >= JWKS_TTL_S
        if stale or (kid not in self._keys and now - self._jwks_at >= JWKS_MIN_REFRESH_S):
            await self._refresh_jwks()
        key = self._keys.get(kid)
        if key is None:
            raise OidcError("ID token signed with an unknown key")
        return key

    async def verify_id_token(self, token: str, nonce: str) -> dict:
        """The verified claims. ``email_verified`` is not enforced here; the caller checks it."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise OidcError("ID token is malformed") from None
        kid = header.get("kid")
        if header.get("alg") != ALGORITHM or not isinstance(kid, str):
            raise OidcError("ID token has an unexpected algorithm or no kid")
        key = await self._signing_key(kid)
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[ALGORITHM],
                audience=self.client_id,
                options={
                    "require": ["iss", "aud", "sub", "exp", "iat", "email", "nonce"],
                    "verify_exp": False,  # checked below against the injectable clock
                    "verify_iat": False,
                    "verify_nbf": False,
                    "verify_iss": False,  # two accepted values, checked below
                },
            )
        except jwt.PyJWTError as exc:
            raise OidcError(f"ID token rejected ({type(exc).__name__})") from None
        now = self._now()
        if claims.get("iss") not in ISSUERS:
            raise OidcError("ID token has the wrong issuer")
        aud = claims["aud"]
        azp = claims.get("azp")
        if (isinstance(aud, list) and len(aud) > 1 and azp is None) or (azp is not None and azp != self.client_id):
            raise OidcError("ID token has the wrong authorized party")
        exp, iat = claims.get("exp"), claims.get("iat")
        if not isinstance(exp, (int, float)) or not isinstance(iat, (int, float)):
            raise OidcError("ID token times are not numbers")
        if exp + LEEWAY_S < now:
            raise OidcError("ID token expired")
        if iat - LEEWAY_S > now:
            raise OidcError("ID token issued in the future")
        got = claims.get("nonce")
        if not isinstance(got, str) or not hmac.compare_digest(got.encode(), nonce.encode()):
            raise OidcError("ID token nonce does not match")
        if not isinstance(claims.get("email"), str) or not claims["email"]:
            raise OidcError("ID token has no email")
        return claims
