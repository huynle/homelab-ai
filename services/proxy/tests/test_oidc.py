"""Google OIDC building blocks: PKCE, the signed login cookie, `next` checks, code exchange, ID token checks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest

import oidc
from fakegoogle import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, TOKEN_URL, FakeGoogle, new_rsa_key

KEY = b"k" * 32 + b"-session-key-for-tests"


class Clock:
    def __init__(self, t: float = 1_793_640_000.0):  # 2026-11-02T17:20:00Z
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def google(clock):
    return FakeGoogle(now=clock)


@pytest.fixture
def client(google, clock):
    return oidc.GoogleOIDC(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, transport=google.transport, now=clock)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- PKCE, state, next


def test_pkce_pair_is_s256():
    verifier, challenge = oidc.pkce_pair()
    assert re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", verifier)
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected
    assert oidc.pkce_pair()[0] != verifier


def test_new_login_is_random():
    a, b = oidc.new_login("/admin/keys"), oidc.new_login(None)
    assert a.next == "/admin/keys" and b.next == "/admin"
    assert len({a.state, b.state, a.nonce, b.nonce}) == 4
    assert all(len(v) >= 32 for v in (a.state, a.nonce, a.verifier))


@pytest.mark.parametrize(
    "value", ["/admin", "/admin/", "/admin/keys", "/admin?x=1", "/admin#top", "/admin/members?a=b"]
)
def test_safe_next_accepts_admin_paths(value):
    assert oidc.safe_next(value) == value


@pytest.mark.parametrize(
    "value",
    [
        None, "", "/", "/adminx", "/v1/models", "//evil.example", "https://evil.example/admin",
        "/admin\\evil", "/admin/\\\\evil", "admin", "/admin/" + "a" * 600, "/admin/\nx", "/admin/ x", 7,
    ],
)  # fmt: skip
def test_safe_next_falls_back_to_admin(value):
    assert oidc.safe_next(value) == "/admin"


def test_login_cookie_round_trip(clock):
    login = oidc.new_login("/admin/keys")
    value = oidc.sign_login(KEY, login, now=clock())
    assert oidc.read_login(KEY, value, now=clock() + 599) == login


def test_login_cookie_rejects_tampering_wrong_key_expiry_and_garbage(clock):
    login = oidc.new_login("/admin")
    value = oidc.sign_login(KEY, login, now=clock())
    payload, sig = value.split(".")
    forged = oidc.sign_login(KEY, oidc.new_login("/admin"), now=clock()).split(".")[0] + "." + sig
    assert oidc.read_login(KEY, forged, now=clock()) is None
    assert oidc.read_login(b"x" * 40, value, now=clock()) is None
    assert oidc.read_login(KEY, value, now=clock() + oidc.LOGIN_TTL_S + 1) is None
    for junk in (None, "", "abc", "a.b", "....", payload, value + "x"):
        assert oidc.read_login(KEY, junk, now=clock()) is None


def test_authorization_url(client):
    login = oidc.new_login("/admin")
    url = urlsplit(client.authorization_url(login))
    assert f"{url.scheme}://{url.netloc}{url.path}" == "https://accounts.google.com/o/oauth2/v2/auth"
    q = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert q == {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": login.state,
        "nonce": login.nonce,
        "code_challenge": oidc.pkce_challenge(login.verifier),
        "code_challenge_method": "S256",
        "prompt": "select_account",
    }


# ---------------------------------------------------------------- code exchange


def test_exchange_posts_pkce_verifier_and_client_credentials(client, google):
    verifier, challenge = oidc.pkce_pair()
    code = google.issue_code(challenge, nonce="n1")
    token = run(client.exchange(code, verifier))
    assert jwt.decode(token, options={"verify_signature": False})["nonce"] == "n1"
    [form] = google.token_requests
    assert form == {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": verifier,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
    }


def test_exchange_failures_raise(client, google):
    verifier, challenge = oidc.pkce_pair()
    with pytest.raises(oidc.OidcError):  # wrong verifier
        run(client.exchange(google.issue_code(challenge), "x" * 43))
    with pytest.raises(oidc.OidcError):  # unknown code
        run(client.exchange("nope", verifier))
    google.token_status = 500
    with pytest.raises(oidc.OidcError):
        run(client.exchange(google.issue_code(), verifier))
    google.token_status = 200
    google.fail_network = True
    with pytest.raises(oidc.OidcError):
        run(client.exchange(google.issue_code(), verifier))


def test_exchange_rejects_reply_without_id_token(clock):
    def handler(request):
        assert str(request.url) == TOKEN_URL
        return httpx.Response(200, json={"access_token": "x"})

    c = oidc.GoogleOIDC(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, transport=httpx.MockTransport(handler), now=clock)
    with pytest.raises(oidc.OidcError):
        run(c.exchange("code", "v" * 43))


def test_http_timeouts_are_set(client):
    assert client.timeout.connect is not None and client.timeout.read is not None
    assert client.timeout.read <= 15


# ---------------------------------------------------------------- ID token checks


def test_verify_happy_path(client, google):
    claims = run(client.verify_id_token(google.id_token(nonce="n1", email="Alice@Example.com"), "n1"))
    assert claims["email"] == "Alice@Example.com" and claims["email_verified"] is True


def test_verify_accepts_bare_issuer(client, google):
    assert run(client.verify_id_token(google.id_token(nonce="n", iss="accounts.google.com"), "n"))


def test_verify_returns_unverified_email_flag(client, google):
    claims = run(client.verify_id_token(google.id_token(nonce="n", email_verified=False), "n"))
    assert claims["email_verified"] is False


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "someone-else.apps.googleusercontent.com"},
        {"aud": [CLIENT_ID, "other"], "azp": "other"},
        {"azp": "other-client"},
        {"iss": "https://evil.example"},
        {"iss": "http://accounts.google.com"},
        {"nonce": "other-nonce"},
        {"nonce": None},
        {"nonce": "n\u00e9"},  # non-ASCII must not crash the comparison
        {"exp": 1_793_640_000 - 61},  # expired beyond the leeway
        {"iat": 1_793_640_000 + 3600},  # issued in the future
        {"exp": None},
        {"iat": None},
        {"email": None},
        {"sub": None},
    ],
)
def test_verify_rejects_bad_claims(client, google, claims):
    token = google.id_token(**{"nonce": "n1", **claims})
    with pytest.raises(oidc.OidcError):
        run(client.verify_id_token(token, "n1"))


def test_verify_allows_small_clock_skew(client, google, clock):
    token = google.id_token(nonce="n", exp=int(clock()) - 30, iat=int(clock()) + 30)
    assert run(client.verify_id_token(token, "n"))


def test_verify_rejects_bad_signature_and_algorithms(client, google):
    forged = google.id_token(nonce="n", key=new_rsa_key())  # right kid, wrong key
    with pytest.raises(oidc.OidcError):
        run(client.verify_id_token(forged, "n"))
    hs = jwt.encode({"nonce": "n", "aud": CLIENT_ID}, "secret" * 8, algorithm="HS256", headers={"kid": google.kid})
    with pytest.raises(oidc.OidcError):
        run(client.verify_id_token(hs, "n"))
    unsigned = jwt.encode({"nonce": "n"}, None, algorithm="none", headers={"kid": google.kid})
    with pytest.raises(oidc.OidcError):
        run(client.verify_id_token(unsigned, "n"))
    for junk in ("", "abc", "a.b.c"):
        with pytest.raises(oidc.OidcError):
            run(client.verify_id_token(junk, "n"))


def test_jwks_is_cached(client, google):
    run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    assert google.jwks_fetches == 1


def test_jwks_refetched_after_ttl(client, google, clock):
    run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    clock.t += oidc.JWKS_TTL_S + 1
    run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    assert google.jwks_fetches == 2


def test_unknown_kid_refreshes_jwks(client, google, clock):
    run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    clock.t += oidc.JWKS_MIN_REFRESH_S + 1
    google.rotate("kid-2")
    assert run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    assert google.jwks_fetches == 2


def test_unknown_kid_refresh_is_rate_limited(client, google):
    run(client.verify_id_token(google.id_token(nonce="n"), "n"))
    for _ in range(3):  # a stream of tokens with a bogus kid must not hammer Google
        with pytest.raises(oidc.OidcError):
            run(client.verify_id_token(google.id_token(nonce="n", kid="bogus"), "n"))
    assert google.jwks_fetches == 1


def test_jwks_fetch_failure_is_an_oidc_error(client, google):
    google.fail_network = True
    with pytest.raises(oidc.OidcError):
        run(client.verify_id_token(google.id_token(nonce="n"), "n"))
