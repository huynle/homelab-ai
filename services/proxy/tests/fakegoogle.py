"""A fake Google OIDC provider for tests: a local RSA key, its JWKS and a token endpoint (httpx.MockTransport).

No network: point :class:`oidc.GoogleOIDC` (or ``create_app(oauth_transport=...)``) at ``FakeGoogle.transport``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import parse_qs

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

CLIENT_ID = "test-client.apps.googleusercontent.com"
CLIENT_SECRET = "test-client-secret-not-real"
PUBLIC_URL = "https://decide.ai.huynle.com"
REDIRECT_URI = PUBLIC_URL + "/admin/auth/callback"
TOKEN_URL = "https://oauth2.googleapis.com/token"
JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"


def new_rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwk_of(private_key, kid: str) -> dict:
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return jwk


class FakeGoogle:
    """Issues ID tokens for codes it handed out, checks PKCE, client credentials and redirect_uri."""

    def __init__(self, now=time.time):
        self.now = now
        self.key = new_rsa_key()
        self.kid = "kid-1"
        self.published = {self.kid: self.key}  # what the JWKS endpoint serves
        self.jwks_fetches = 0
        self.token_requests: list[dict] = []
        self.codes: dict[str, dict] = {}  # code -> {"claims": ..., "challenge": ...}
        self.token_status = 200
        self.fail_network = False

    # ---- tokens

    def id_token(self, *, key=None, kid=None, alg="RS256", **claims) -> str:
        now = int(self.now())
        body = {
            "iss": "https://accounts.google.com",
            "aud": CLIENT_ID,
            "azp": CLIENT_ID,
            "sub": "1234567890",
            "email": "owner@example.com",
            "email_verified": True,
            "iat": now,
            "exp": now + 3600,
        }
        body.update(claims)
        body = {k: v for k, v in body.items() if v is not None}
        headers = {"kid": kid or self.kid}
        return jwt.encode(body, key or self.key, algorithm=alg, headers=headers)

    def rotate(self, kid: str = "kid-2", keep_old: bool = False):
        self.key = new_rsa_key()
        self.kid = kid
        if not keep_old:
            self.published = {}
        self.published[kid] = self.key

    def issue_code(self, challenge: str | None = None, **claims) -> str:
        """A code the token endpoint will exchange for an ID token with ``claims``."""
        code = "code-" + secrets.token_urlsafe(8)
        self.codes[code] = {"claims": claims, "challenge": challenge}
        return code

    # ---- HTTP

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail_network:
            raise httpx.ConnectError("no route to Google", request=request)
        url = str(request.url).split("?")[0]
        if url == JWKS_URL and request.method == "GET":
            self.jwks_fetches += 1
            return httpx.Response(200, json={"keys": [jwk_of(k, kid) for kid, k in self.published.items()]})
        if url == TOKEN_URL and request.method == "POST":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_requests.append(form)
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "server_error"})
            entry = self.codes.pop(form.get("code", ""), None)
            ok = (
                entry is not None
                and form.get("grant_type") == "authorization_code"
                and form.get("client_id") == CLIENT_ID
                and form.get("client_secret") == CLIENT_SECRET
                and form.get("redirect_uri") == REDIRECT_URI
                and (entry["challenge"] is None or s256(form.get("code_verifier", "")) == entry["challenge"])
            )
            if not ok:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(
                200,
                json={"access_token": "ya29.fake", "token_type": "Bearer", "expires_in": 3599,
                      "id_token": self.id_token(**entry["claims"])},
            )  # fmt: skip
        raise AssertionError(f"unexpected request to {request.url}")

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def s256(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
