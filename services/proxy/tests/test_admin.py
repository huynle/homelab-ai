"""The admin page: activation, Google sign-in, sessions, CSRF, the JSON API, HTML pages and isolation from /v1."""

from __future__ import annotations

import hashlib
import json
import logging
import re

import httpx
import pytest
from fastapi.testclient import TestClient

import keyservice as ks
from adminkit import API_ROUTES, BODIES, ORIGIN, SESSION_KEY, UNSAFE_ROUTES, admin_settings, set_cookie_headers
from app import create_app
from config import Settings
from conftest import ALICE, API_KEY, BOB, OWNER, PEPPER
from fakegoogle import CLIENT_ID, PUBLIC_URL, REDIRECT_URI

SESSION = "__Host-decision_session"
LOGIN = "__Host-decision_oauth"
MALLORY = "mallory@example.com"


def audit(db, event=None):
    rows = db.conn.execute("SELECT event, actor_email, target_email, detail FROM audit ORDER BY id").fetchall()
    out = [(r["event"], r["actor_email"], r["target_email"], json.loads(r["detail"])) for r in rows]
    return [e for e in out if event is None or e[0] == event]


def invite(admin, email):
    ks.invite_member(admin.db, ks.get_viewer(admin.db, OWNER), email, now=admin.clock())


def err(resp):
    return resp.json()["error"]["code"]


# ---------------------------------------------------------------- activation (production default stays inert)


ADMIN_PATHS = ["/admin", "/admin/", "/admin/auth/login", "/admin/auth/callback?state=x", "/admin/api/me",
               "/admin/api/keys", "/admin/static/admin.js", "/admin/members"]  # fmt: skip


def test_default_file_mode_has_no_admin(client, auth):
    for path in ADMIN_PATHS:
        for headers in ({}, auth):
            resp = client.get(path, headers=headers)
            assert resp.status_code == 404, path
            assert "Sign in" not in resp.text and "location" not in resp.headers
    assert client.post("/admin/api/keys", json={"name": "x"}, headers=auth).status_code == 404
    assert client.post("/admin/auth/logout").status_code == 404


def test_default_file_mode_v1_unchanged(client, auth):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=auth).status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/v1/health").status_code == 200
    assert client.get("/administrator", headers={}).status_code == 401  # not an admin path
    client.get("/admin")
    mx = client.app.state.metrics.registry
    assert mx.get_sample_value(
        "decision_api_requests_total", {"route": "other", "model": "none", "code": "404", "key_id": "none"}
    ) == 1  # fmt: skip


def test_file_mode_ignores_oauth_settings(tmp_path, backends):
    s = admin_settings(tmp_path, keystore="file")
    app = create_app(s, transport=httpx.MockTransport(backends.handler))
    with TestClient(app, base_url=PUBLIC_URL) as c:
        assert c.get("/admin").status_code == 404
        assert c.get("/admin/auth/login").status_code == 404
        assert c.get("/v1/models").status_code == 401


def test_sqlite_mode_without_oauth_has_no_admin(tmp_path, backends, caplog):
    s = admin_settings(tmp_path, oauth_client_id=None, oauth_client_secret_file=None, public_url=None,
                       session_key_file=None)  # fmt: skip
    caplog.set_level(logging.INFO, logger="decision_api")
    app = create_app(s, transport=httpx.MockTransport(backends.handler))
    with TestClient(app, base_url=PUBLIC_URL) as c:
        for path in ADMIN_PATHS:
            assert c.get(path).status_code == 404, path
        assert c.get("/v1/models").status_code == 401
    assert "admin page disabled" in caplog.text
    app.state.db.close()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"oauth_client_id": None}, "DECISION_API_OAUTH_CLIENT_ID"),
        ({"oauth_client_secret_file": None}, "DECISION_API_OAUTH_CLIENT_SECRET_FILE"),
        ({"oauth_client_secret_file": "/nonexistent/secret"}, "DECISION_API_OAUTH_CLIENT_SECRET_FILE"),
        ({"public_url": None}, "DECISION_API_PUBLIC_URL"),
        ({"public_url": "http://decide.ai.huynle.com"}, "DECISION_API_PUBLIC_URL"),
        ({"public_url": "https://decide.ai.huynle.com/admin"}, "DECISION_API_PUBLIC_URL"),
        ({"public_url": "https://decide.ai.huynle.com?x=1"}, "DECISION_API_PUBLIC_URL"),
        ({"public_url": "https://user@decide.ai.huynle.com"}, "DECISION_API_PUBLIC_URL"),
        ({"session_key_file": None}, "DECISION_API_SESSION_KEY_FILE"),
        ({"session_key_file": "/nonexistent/key"}, "DECISION_API_SESSION_KEY_FILE"),
    ],
)
def test_partial_or_invalid_oauth_config_refuses_to_start(tmp_path, change, match):
    with pytest.raises(RuntimeError, match=match):
        create_app(admin_settings(tmp_path, **change))


def test_short_or_reused_session_key_refuses_to_start(tmp_path):
    short = tmp_path / "short"
    short.write_bytes(b"x" * 31)
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        create_app(admin_settings(tmp_path, session_key_file=str(short)))
    same = tmp_path / "same"
    same.write_bytes(PEPPER)
    with pytest.raises(RuntimeError, match="must differ"):
        create_app(admin_settings(tmp_path, session_key_file=str(same)))


def test_oauth_settings_from_env():
    s = Settings.from_env(
        {
            "DECISION_API_KEYS": "k",
            "DECISION_API_OAUTH_CLIENT_ID": " id.apps.googleusercontent.com ",
            "DECISION_API_OAUTH_CLIENT_SECRET_FILE": "/run/secret",
            "DECISION_API_PUBLIC_URL": "https://decide.ai.huynle.com/",
            "DECISION_API_SESSION_KEY_FILE": "/run/session-key",
        }
    )
    assert s.oauth_client_id == "id.apps.googleusercontent.com"
    assert s.oauth_client_secret_file == "/run/secret"
    assert s.public_url == "https://decide.ai.huynle.com/"
    assert s.session_key_file == "/run/session-key"
    d = Settings.from_env({"DECISION_API_KEYS": "k"})
    assert (d.oauth_client_id, d.oauth_client_secret_file, d.public_url, d.session_key_file) == (None,) * 4


def test_secrets_not_in_repr(admin):
    assert "test-client-secret" not in repr(admin.app.state.admin)
    assert SESSION_KEY.decode() not in repr(admin.app.state.admin)


# ---------------------------------------------------------------- login


def test_login_redirects_to_google_with_pkce_state_nonce(admin):
    c = admin.client()
    resp = c.get("/admin/auth/login", params={"next": "/admin/keys"})
    assert resp.status_code == 302
    loc = resp.headers["location"]
    assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    q = admin.start_login(c)
    assert q["client_id"] == CLIENT_ID and q["redirect_uri"] == REDIRECT_URI
    assert q["response_type"] == "code" and q["scope"] == "openid email profile"
    assert q["code_challenge_method"] == "S256" and re.fullmatch(r"[A-Za-z0-9_-]{43}", q["code_challenge"])
    assert q["prompt"] == "select_account"
    assert len(q["state"]) >= 32 and len(q["nonce"]) >= 32 and q["state"] != q["nonce"]
    [cookie] = set_cookie_headers(resp, LOGIN)
    attrs = {a.strip().lower() for a in cookie.split(";")[1:]}
    assert {"httponly", "secure", "path=/", "samesite=lax", "max-age=600"} <= attrs
    assert not any(a.startswith("domain") for a in attrs)
    assert q["state"] not in loc.split("state=")[0]  # sanity
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "next_,expected",
    [("/admin?tab=usage", "/admin?tab=usage"), ("https://evil.example/", "/admin"), ("//evil.example", "/admin"),
     (None, "/admin")],
)  # fmt: skip
def test_next_is_same_origin_admin_only(admin, next_, expected):
    resp = admin.sign_in(admin.client(), OWNER, next_)
    assert resp.status_code == 302
    assert resp.headers["location"] == expected


# ---------------------------------------------------------------- callback


def test_callback_owner_happy_path(admin):
    c = admin.client()
    resp = admin.sign_in(c, OWNER)
    assert resp.status_code == 302 and resp.headers["location"] == "/admin"
    [cookie] = set_cookie_headers(resp, SESSION)
    value = cookie.split(";")[0].split("=", 1)[1]
    attrs = {a.strip().lower() for a in cookie.split(";")[1:]}
    assert {"httponly", "secure", "path=/", "samesite=lax", "max-age=43200"} <= attrs
    assert not any(a.startswith("domain") for a in attrs)
    [cleared] = set_cookie_headers(resp, LOGIN)
    assert "max-age=0" in cleared.lower()
    me = c.get("/admin/api/me")
    assert me.status_code == 200
    assert me.json()["email"] == OWNER and me.json()["role"] == "owner" and len(me.json()["csrf_token"]) >= 32
    # only hashes are stored
    row = admin.db.conn.execute("SELECT * FROM sessions").fetchone()
    assert row["id_hash"] == hashlib.sha256(value.encode()).hexdigest()
    assert row["csrf_token_hash"] == hashlib.sha256(me.json()["csrf_token"].encode()).hexdigest()
    assert row["expires_at"] == "2026-11-03T06:00:00Z"
    dump = "\n".join(admin.db.conn.iterdump())
    assert value not in dump and me.json()["csrf_token"] not in dump
    assert audit(admin.db, "sign_in.allowed") == [("sign_in.allowed", OWNER, None, {"role": "owner"})]
    assert ks.list_members(admin.db, ks.get_viewer(admin.db, OWNER))[0]["last_sign_in_at"] == "2026-11-02T18:00:00Z"


def test_callback_member_happy_path_case_insensitive(admin):
    invite(admin, ALICE)
    c = admin.client()
    resp = admin.sign_in(c, "Alice@Example.COM", "/admin?x=1")
    assert resp.status_code == 302 and resp.headers["location"] == "/admin?x=1"
    assert c.get("/admin/api/me").json()["email"] == ALICE
    assert c.get("/admin/api/me").json()["role"] == "member"
    assert audit(admin.db, "sign_in.allowed") == [("sign_in.allowed", ALICE, None, {"role": "member"})]


def _denied(admin, resp, status, reason, actor=None):
    assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("text/html")
    assert not set_cookie_headers(resp, SESSION) or "max-age=0" in set_cookie_headers(resp, SESSION)[0].lower()
    assert admin.db.conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    assert audit(admin.db, "sign_in.denied")[-1] == ("sign_in.denied", actor, None, {"reason": reason})


def test_callback_bad_state(admin):
    c = admin.client()
    q = admin.start_login(c)
    code = admin.google.issue_code(q["code_challenge"], email=OWNER, nonce=q["nonce"])
    resp = c.get("/admin/auth/callback", params={"code": code, "state": q["state"] + "x"})
    _denied(admin, resp, 400, "state_mismatch")
    assert admin.google.token_requests == []  # no exchange on a bad state
    assert "max-age=0" in set_cookie_headers(resp, LOGIN)[0].lower()


def test_callback_missing_login_cookie(admin):
    c = admin.client()
    q = admin.start_login(c)
    c.cookies.clear()
    code = admin.google.issue_code(q["code_challenge"], email=OWNER, nonce=q["nonce"])
    _denied(admin, c.get("/admin/auth/callback", params={"code": code, "state": q["state"]}), 400, "state_mismatch")


def test_callback_expired_login_cookie(admin):
    c = admin.client()
    q = admin.start_login(c)
    admin.clock.advance(seconds=601)
    code = admin.google.issue_code(q["code_challenge"], email=OWNER, nonce=q["nonce"])
    _denied(admin, c.get("/admin/auth/callback", params={"code": code, "state": q["state"]}), 400, "state_mismatch")


def test_callback_replay_fails(admin):
    c = admin.client()
    q = admin.start_login(c)
    code = admin.google.issue_code(q["code_challenge"], email=OWNER, nonce=q["nonce"])
    params = {"code": code, "state": q["state"]}
    assert c.get("/admin/auth/callback", params=params).status_code == 302
    assert c.get("/admin/auth/callback", params=params).status_code == 400  # login cookie was cleared


def test_callback_google_error(admin):
    c = admin.client()
    q = admin.start_login(c)
    resp = c.get("/admin/auth/callback", params={"error": "access_denied", "state": q["state"]})
    _denied(admin, resp, 403, "google_error")


def test_callback_missing_code(admin):
    c = admin.client()
    q = admin.start_login(c)
    resp = c.get("/admin/auth/callback", params={"state": q["state"]})
    assert resp.status_code == 400
    assert admin.db.conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_callback_token_endpoint_failure(admin):
    admin.google.token_status = 500
    _denied(admin, admin.sign_in(admin.client(), OWNER), 400, "token_invalid")


@pytest.mark.parametrize(
    "claims",
    [
        {"nonce": "wrong-nonce"},
        {"aud": "someone-else.apps.googleusercontent.com"},
        {"iss": "https://evil.example"},
        {"exp": 1_793_642_400 - 3600, "iat": 1_793_642_400 - 7200},  # expired an hour ago
    ],
)
def test_callback_rejects_invalid_id_token(admin, claims):
    _denied(admin, admin.sign_in(admin.client(), OWNER, **claims), 400, "token_invalid")


def test_callback_email_unverified(admin):
    resp = admin.sign_in(admin.client(), OWNER, email_verified=False)
    _denied(admin, resp, 403, "email_unverified", actor=OWNER)


def test_callback_not_invited(admin):
    resp = admin.sign_in(admin.client(), MALLORY)
    _denied(admin, resp, 403, "not_invited", actor=MALLORY)
    assert "not allowed" in resp.text.lower() and "invite" in resp.text.lower()
    assert "not_invited" not in resp.text


def test_callback_removed_member(admin):
    invite(admin, ALICE)
    c = admin.signed_in(ALICE)
    ks.remove_member(admin.db, ks.get_viewer(admin.db, OWNER), ALICE, now=admin.clock())
    assert c.get("/admin/api/me").status_code == 401  # existing session ended
    _denied(admin, admin.sign_in(admin.client(), ALICE), 403, "not_invited", actor=ALICE)


def test_callback_unknown_kid_refreshes_jwks(admin):
    admin.signed_in(OWNER)
    assert admin.google.jwks_fetches == 1
    admin.clock.advance(minutes=5)
    admin.google.rotate("kid-2")
    assert admin.sign_in(admin.client(), OWNER).status_code == 302
    assert admin.google.jwks_fetches == 2


# ---------------------------------------------------------------- sessions


def test_session_expires_after_12_hours_not_sliding(admin):
    c = admin.signed_in(OWNER)
    admin.clock.advance(hours=11, minutes=59)
    assert c.get("/admin/api/me").status_code == 200
    admin.clock.advance(minutes=1, seconds=1)
    resp = c.get("/admin/api/me")
    assert resp.status_code == 401 and err(resp) == "unauthorized"


def test_session_needs_an_active_user(admin):
    """Defense in depth: a session row alone is not enough once the user is removed (or was never invited)."""
    invite(admin, ALICE)
    c = admin.signed_in(ALICE)
    admin.db.conn.execute("UPDATE users SET removed_at = '2026-11-02T18:00:00Z' WHERE email = ?", (ALICE,))
    assert c.get("/admin/api/me").status_code == 401
    assert c.get("/admin").status_code == 200 and "Sign in with Google" in c.get("/admin").text


def test_role_comes_from_the_users_table_not_the_session(admin):
    invite(admin, ALICE)
    c = admin.signed_in(ALICE)
    assert c.get("/admin/api/members").status_code == 403
    ks.ensure_owners(admin.db, [OWNER, ALICE], now=admin.clock())  # promoted in config: takes effect at once
    assert c.get("/admin/api/members").status_code == 200


def test_rotating_the_session_key_ends_sessions(admin, tmp_path, backends):
    from adminkit import make_admin, write_secret

    c = admin.signed_in(OWNER)
    value = c.cookies.get(SESSION)
    admin.app.state.db.close()
    rotated = make_admin(tmp_path, backends, admin.clock, session_key_file=write_secret(tmp_path / "k2", b"r" * 40))
    try:
        replay = rotated.client()
        replay.cookies.set(SESSION, value, domain="decide.ai.huynle.com")
        assert replay.get("/admin/api/me").status_code == 401
    finally:
        rotated.close()
        admin.app.state.db = rotated.app.state.db  # closed above; keep the fixture teardown harmless


def test_forged_session_cookie(admin):
    c = admin.client()
    c.cookies.set(SESSION, "forged-session-id", domain="decide.ai.huynle.com")
    assert c.get("/admin/api/me").status_code == 401


def test_logout(admin):
    c = admin.signed_in(OWNER)
    value = c.cookies.get(SESSION)
    resp = admin.call(c, "POST", "/admin/auth/logout")
    assert resp.status_code == 204
    assert "max-age=0" in set_cookie_headers(resp, SESSION)[0].lower()
    assert c.get("/admin/api/me").status_code == 401
    replay = admin.client()
    replay.cookies.set(SESSION, value, domain="decide.ai.huynle.com")
    assert replay.get("/admin/api/me").status_code == 401  # revoked server-side
    assert admin.db.conn.execute("SELECT revoked_at FROM sessions").fetchone()[0] is not None


def test_logout_needs_csrf_and_session(admin):
    c = admin.signed_in(OWNER)
    assert err(admin.call(c, "POST", "/admin/auth/logout", csrf=None)) == "csrf_failed"
    assert c.get("/admin/api/me").status_code == 200
    assert admin.client().post("/admin/auth/logout", headers=ORIGIN).status_code == 401


# ---------------------------------------------------------------- auth matrix: every JSON route


@pytest.mark.parametrize("method,path", API_ROUTES)
def test_every_api_route_rejects_signed_out(admin, method, path):
    resp = admin.client().request(method, path, json=BODIES.get((method, path)), headers=ORIGIN)
    assert resp.status_code == 401
    assert resp.json() == {"error": {"code": "unauthorized", "message": "sign in required"}}
    assert "location" not in resp.headers
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("method,path", API_ROUTES)
def test_every_api_route_rejects_bearer_key_only(admin, method, path):
    db = admin.db
    raw = ks.create_key(db, ks.get_viewer(db, OWNER), "owner key", now=admin.clock())["key"]
    for key in (raw, API_KEY):  # an issued key and the imported SOPS key
        headers = {**ORIGIN, "Authorization": f"Bearer {key}"}
        resp = admin.client().request(method, path, json=BODIES.get((method, path)), headers=headers)
        assert resp.status_code == 401 and err(resp) == "unauthorized"


@pytest.mark.parametrize("method,path", API_ROUTES + [("POST", "/admin/auth/logout")])
def test_every_api_route_rejects_cross_site(admin, method, path):
    c = admin.signed_in(OWNER)
    token = admin.csrf(c)
    for origin in ("https://evil.example", "null", "http://decide.ai.huynle.com", "https://decide.ai.huynle.com.evil"):
        headers = {"Origin": origin, "X-CSRF-Token": token}
        resp = c.request(method, path, json=BODIES.get((method, path)), headers=headers)
        assert resp.status_code == 403 and err(resp) == "csrf_failed", (origin, resp.text)


@pytest.mark.parametrize("method,path", UNSAFE_ROUTES)
def test_unsafe_routes_need_csrf_token_and_origin(admin, method, path):
    c = admin.signed_in(OWNER)
    other = admin.signed_in(OWNER)
    body = BODIES.get((method, path))
    cases = [
        {"Origin": PUBLIC_URL},  # no token
        {"Origin": PUBLIC_URL, "X-CSRF-Token": "wrong" * 10},
        {"Origin": PUBLIC_URL, "X-CSRF-Token": admin.csrf(other)},  # another session's token
        {"X-CSRF-Token": admin.csrf(c)},  # no Origin
    ]
    for headers in cases:
        resp = c.request(method, path, json=body, headers=headers)
        assert resp.status_code == 403, (headers, resp.text)
        assert resp.json() == {"error": {"code": "csrf_failed", "message": "missing or invalid CSRF token or Origin"}}
    assert ks.list_keys(admin.db, ks.get_viewer(admin.db, OWNER), now=admin.clock())[0]["name"] == "imported (SOPS)"
    assert c.get("/admin/api/me").status_code == 200  # not signed out


def test_route_inventory_matches_contract(admin):
    """Every /admin route the app serves is either in the contract, an HTML page, or a public auth/static route."""
    served = {(m, r.path) for r in admin.app.routes if getattr(r, "path", "").startswith("/admin") for m in r.methods}
    contract = {
        ("GET", "/admin/api/me"), ("GET", "/admin/api/keys"), ("POST", "/admin/api/keys"),
        ("PATCH", "/admin/api/keys/{id}"), ("POST", "/admin/api/keys/{id}/revoke"), ("GET", "/admin/api/usage"),
        ("GET", "/admin/api/members"), ("POST", "/admin/api/members"), ("DELETE", "/admin/api/members/{email}"),
        ("GET", "/admin/api/audit"), ("GET", "/admin/auth/login"), ("GET", "/admin/auth/callback"),
        ("POST", "/admin/auth/logout"),
    }  # fmt: skip
    pages = {("GET", p) for p in ("/admin", "/admin/keys", "/admin/usage", "/admin/members", "/admin/audit")}
    static = {("GET", "/admin/static/{name}")}
    assert served - {("HEAD", p) for _, p in served} == contract | pages | static


@pytest.mark.parametrize("path", ["/admin/keys", "/admin/usage", "/admin/members", "/admin/audit"])
def test_owner_pages_signed_out_redirect_to_login(admin, path):
    resp = admin.client().get(path)
    assert resp.status_code == 302
    assert resp.headers["location"] == f"/admin/auth/login?next={path.replace('/', '%2F')}"


def test_admin_home_signed_out_shows_only_sign_in(admin):
    ks.create_key(admin.db, ks.get_viewer(admin.db, OWNER), "secret-name", now=admin.clock())
    resp = admin.client().get("/admin", headers={"Authorization": f"Bearer {API_KEY}"})
    assert resp.status_code == 200
    assert 'href="/admin/auth/login"' in resp.text and "Sign in with Google" in resp.text
    assert "secret-name" not in resp.text and "csrf-token" not in resp.text


def test_session_cookie_does_not_authenticate_v1(admin):
    c = admin.signed_in(OWNER)
    resp = c.get("/v1/models")
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "unauthorized"
    assert c.post("/v1/decide/text", json={}, headers=ORIGIN).status_code == 401
    assert c.get("/v1/health").status_code in (200, 503)  # still public
    assert c.get("/healthz").status_code == 200


# ---------------------------------------------------------------- JSON API: keys


def test_me(admin):
    invite(admin, ALICE)
    for email, role in ((OWNER, "owner"), (ALICE, "member")):
        me = admin.signed_in(email).get("/admin/api/me")
        assert me.status_code == 200 and me.headers["cache-control"] == "no-store"
        assert set(me.json()) == {"email", "role", "csrf_token"} and me.json()["role"] == role


def test_create_key_shows_it_once(admin):
    c = admin.signed_in(OWNER)
    resp = admin.call(c, "POST", "/admin/api/keys", {"name": " laptop ", "expires_at": "2027-11-02T00:00:00Z"})
    assert resp.status_code == 201
    body = resp.json()
    assert resp.headers["location"] == f"/admin/api/keys/{body['id']}"
    assert re.fullmatch(r"dk_[a-z2-7]{12}_[A-Za-z0-9_-]{43}", body["key"])
    assert body["name"] == "laptop" and body["owner_email"] == OWNER and body["status"] == "active"
    assert body["expires_at"] == "2027-11-02T00:00:00Z" and body["rate_limit_rpm"] is None
    listed = c.get("/admin/api/keys").json()["keys"]
    assert all("key" not in k for k in listed)
    assert body["key"] not in c.get("/admin/api/keys").text
    assert body["key"] not in c.get("/admin").text
    assert body["key"] not in "\n".join(admin.db.conn.iterdump())
    assert audit(admin.db, "key.created")[-1] == ("key.created", OWNER, OWNER, {"name": "laptop"})


@pytest.mark.parametrize(
    "body,message",
    [
        ({"name": "x", "rate_limit_rpm": 5}, "rate_limit_rpm: unknown field"),
        ({"name": ""}, "name: must be 1 to 64 characters"),
        ({"name": "x" * 65}, "name: must be 1 to 64 characters"),
        ({"name": 3}, "name: must be 1 to 64 characters"),
        ({}, "name: must be 1 to 64 characters"),
        ({"name": "x", "expires_at": "2030-01-01T00:00:00Z"}, "expires_at: must be at most 2 years from now"),
        ({"name": "x", "expires_at": "2020-01-01T00:00:00Z"}, "expires_at: must be in the future"),
        ([1], "request body must be a JSON object"),
    ],
)
def test_create_key_rejects(admin, body, message):
    c = admin.signed_in(OWNER)
    resp = admin.call(c, "POST", "/admin/api/keys", body)
    assert resp.status_code == 400
    assert resp.json() == {"error": {"code": "invalid_request", "message": message}}


def test_create_key_rejects_bad_json_and_huge_bodies(admin):
    c = admin.signed_in(OWNER)
    resp = admin.call(c, "POST", "/admin/api/keys", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400 and err(resp) == "invalid_request"
    resp = admin.call(c, "POST", "/admin/api/keys", {"name": "x" * 100_000})
    assert resp.status_code == 400 and err(resp) == "invalid_request"


def test_member_key_limit(admin):
    invite(admin, ALICE)
    c = admin.signed_in(ALICE)
    token = admin.csrf(c)
    for i in range(20):
        assert admin.call(c, "POST", "/admin/api/keys", {"name": f"k{i}"}, csrf=token).status_code == 201
    resp = admin.call(c, "POST", "/admin/api/keys", {"name": "k21"}, csrf=token)
    assert resp.status_code == 409 and err(resp) == "conflict"


def test_list_keys_owner_vs_member(admin):
    invite(admin, ALICE)
    invite(admin, BOB)
    alice, bob, owner = admin.signed_in(ALICE), admin.signed_in(BOB), admin.signed_in(OWNER)
    a = admin.call(alice, "POST", "/admin/api/keys", {"name": "alice-1"}).json()
    admin.call(bob, "POST", "/admin/api/keys", {"name": "bob-1"})
    assert [k["name"] for k in alice.get("/admin/api/keys").json()["keys"]] == ["alice-1"]
    assert alice.get("/admin/api/keys", params={"user": ALICE}).status_code == 200
    resp = alice.get("/admin/api/keys", params={"user": BOB})
    assert resp.status_code == 403 and err(resp) == "forbidden"
    assert {k["name"] for k in owner.get("/admin/api/keys").json()["keys"]} == {"alice-1", "bob-1", "imported (SOPS)"}
    assert [k["id"] for k in owner.get("/admin/api/keys", params={"user": ALICE}).json()["keys"]] == [a["id"]]
    assert owner.get("/admin/api/keys", params={"user": "not-an-email"}).status_code == 400


def test_rename_key(admin):
    invite(admin, ALICE)
    alice, owner = admin.signed_in(ALICE), admin.signed_in(OWNER)
    key = admin.call(alice, "POST", "/admin/api/keys", {"name": "a"}).json()
    resp = admin.call(alice, "PATCH", f"/admin/api/keys/{key['id']}", {"name": "renamed"})
    assert resp.status_code == 200 and resp.json()["name"] == "renamed" and "key" not in resp.json()
    assert admin.call(alice, "PATCH", f"/admin/api/keys/{key['id']}", {}).json()["name"] == "renamed"
    resp = admin.call(alice, "PATCH", f"/admin/api/keys/{key['id']}", {"name": "x", "owner_email": OWNER})
    assert resp.status_code == 400 and resp.json()["error"]["message"] == "owner_email: unknown field"
    assert admin.call(owner, "PATCH", f"/admin/api/keys/{key['id']}", {"name": "by-owner"}).status_code == 200
    imported = owner.get("/admin/api/keys", params={"user": OWNER}).json()["keys"][0]
    resp = admin.call(alice, "PATCH", f"/admin/api/keys/{imported['id']}", {"name": "mine"})
    assert resp.status_code == 404 and err(resp) == "not_found"
    assert admin.call(alice, "PATCH", "/admin/api/keys/NOT-A-KEY-ID", {"name": "x"}).status_code == 404
    admin.call(alice, "POST", f"/admin/api/keys/{key['id']}/revoke")
    resp = admin.call(alice, "PATCH", f"/admin/api/keys/{key['id']}", {"name": "again"})
    assert resp.status_code == 409 and err(resp) == "conflict"


def test_revoke_key(admin):
    invite(admin, ALICE)
    alice, owner = admin.signed_in(ALICE), admin.signed_in(OWNER)
    key = admin.call(alice, "POST", "/admin/api/keys", {"name": "a"}).json()
    imported = owner.get("/admin/api/keys", params={"user": OWNER}).json()["keys"][0]
    assert admin.call(alice, "POST", f"/admin/api/keys/{imported['id']}/revoke").status_code == 404
    first = admin.call(alice, "POST", f"/admin/api/keys/{key['id']}/revoke")
    assert first.status_code == 200 and first.json()["status"] == "revoked"
    admin.clock.advance(minutes=1)
    again = admin.call(owner, "POST", f"/admin/api/keys/{key['id']}/revoke")
    assert again.status_code == 200 and again.json()["revoked_at"] == first.json()["revoked_at"]
    assert len(audit(admin.db, "key.revoked")) == 1
    resp = admin.call(alice, "POST", f"/admin/api/keys/{key['id']}/revoke", {"reason": "x"})
    assert resp.status_code == 400 and resp.json()["error"]["message"] == "reason: unknown field"


# ---------------------------------------------------------------- JSON API: usage, members, audit


def test_usage_owner_vs_member(admin):
    invite(admin, ALICE)
    invite(admin, BOB)
    alice, owner = admin.signed_in(ALICE), admin.signed_in(OWNER)
    bob_key = ks.create_key(admin.db, ks.get_viewer(admin.db, BOB), "b", now=admin.clock())
    resp = alice.get("/admin/api/usage")
    assert resp.status_code == 200
    assert resp.json()["from"] == "2026-10-04" and resp.json()["to"] == "2026-11-02"
    assert alice.get("/admin/api/usage", params={"user": BOB}).status_code == 403
    assert alice.get("/admin/api/usage", params={"key_id": bob_key["id"]}).status_code == 404
    assert owner.get("/admin/api/usage", params={"key_id": bob_key["id"], "group_by": "key"}).status_code == 200
    for bad in ({"group_by": "hour"}, {"from": "2026-11-05", "to": "2026-11-01"}, {"from": "yesterday"}):
        resp = owner.get("/admin/api/usage", params=bad)
        assert resp.status_code == 400 and err(resp) == "invalid_request"


def test_members_api(admin):
    invite(admin, ALICE)
    owner, alice = admin.signed_in(OWNER), admin.signed_in(ALICE)
    resp = owner.get("/admin/api/members")
    assert [m["email"] for m in resp.json()["members"]] == [OWNER, ALICE]
    assert alice.get("/admin/api/members").status_code == 403
    assert err(admin.call(alice, "POST", "/admin/api/members", {"email": "x@example.com"})) == "forbidden"
    resp = admin.call(owner, "POST", "/admin/api/members", {"email": "Bob@Example.com"})
    assert resp.status_code == 201 and resp.json()["email"] == BOB
    assert resp.headers["location"] == "/admin/api/members/bob@example.com"
    assert admin.call(owner, "POST", "/admin/api/members", {"email": BOB, "role": "member"}).status_code == 409
    assert admin.call(owner, "POST", "/admin/api/members", {"email": OWNER}).status_code == 409
    for bad in ({"email": "carol@example.com", "role": "owner"}, {"email": "nope"}, {"email": "c@example.com", "x": 1}):
        assert admin.call(owner, "POST", "/admin/api/members", bad).status_code == 400, bad


def test_remove_member_revokes_keys_and_sessions(admin):
    invite(admin, ALICE)
    owner, alice = admin.signed_in(OWNER), admin.signed_in(ALICE)
    key = admin.call(alice, "POST", "/admin/api/keys", {"name": "a"}).json()
    assert err(admin.call(alice, "DELETE", "/admin/api/members/alice@example.com")) == "forbidden"
    resp = admin.call(owner, "DELETE", "/admin/api/members/ALICE%40example.com")
    assert resp.status_code == 200 and resp.json() == {"email": ALICE, "revoked_keys": [key["id"]]}
    assert alice.get("/admin/api/me").status_code == 401
    assert owner.get("/admin/api/keys", params={"user": ALICE}).json()["keys"][0]["status"] == "revoked"
    resp = admin.call(owner, "DELETE", f"/admin/api/members/{OWNER}")
    assert resp.status_code == 403 and resp.json()["error"]["message"] == "the owner cannot be removed"
    assert admin.call(owner, "DELETE", "/admin/api/members/nobody@example.com").status_code == 404


def test_audit_api(admin):
    invite(admin, ALICE)
    owner, alice = admin.signed_in(OWNER), admin.signed_in(ALICE)
    assert alice.get("/admin/api/audit").status_code == 403
    page = owner.get("/admin/api/audit", params={"limit": 2}).json()
    assert len(page["entries"]) == 2 and page["next_before"] == page["entries"][-1]["id"]
    rest = owner.get("/admin/api/audit", params={"before": page["next_before"]}).json()
    assert [e["event"] for e in rest["entries"]][-1] == "key.created"  # the SOPS import
    assert rest["next_before"] is None
    for bad in ({"limit": "abc"}, {"limit": 0}, {"limit": 501}, {"before": "-1"}, {"to": "2026-13-01"}):
        resp = owner.get("/admin/api/audit", params=bad)
        assert resp.status_code == 400 and err(resp) == "invalid_request", bad


# ---------------------------------------------------------------- full flow and isolation from /v1


def test_full_flow_invite_sign_in_key_use_usage_revoke(admin, laya_request):
    owner = admin.signed_in(OWNER)
    assert admin.call(owner, "POST", "/admin/api/members", {"email": BOB}).status_code == 201
    bob = admin.signed_in(BOB)
    created = admin.call(bob, "POST", "/admin/api/keys", {"name": "n8n"}).json()
    api = admin.client()
    auth = {"Authorization": f"Bearer {created['key']}"}
    assert api.get("/v1/models", headers=auth).status_code == 200
    assert api.post("/v1/decide/text", json=laya_request, headers=auth).status_code == 200
    for _ in range(3):  # admin traffic is never usage
        bob.get("/admin/api/keys", headers=auth)
        bob.get("/admin", headers=auth)
    assert admin.app.state.usage.flush()
    usage = bob.get("/admin/api/usage", params={"group_by": "route"}).json()
    assert {r["route"]: r["requests"] for r in usage["rows"]} == {"/v1/models": 1, "/v1/decide/text": 1}
    assert usage["total"]["requests"] == 2
    assert owner.get("/admin/api/usage", params={"group_by": "key"}).json()["rows"][0]["key_id"] == created["id"]
    routes = {r[0] for r in admin.db.conn.execute("SELECT route FROM usage_daily")}
    assert routes == {"/v1/models", "/v1/decide/text"}
    assert bob.get("/admin/api/keys").json()["keys"][0]["last_used_at"] is not None
    assert admin.call(bob, "POST", f"/admin/api/keys/{created['id']}/revoke").status_code == 200
    resp = api.get("/v1/models", headers=auth)
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "unauthorized"


def test_admin_traffic_metrics_are_bounded(admin):
    c = admin.signed_in(OWNER)
    key = admin.call(c, "POST", "/admin/api/keys", {"name": "m"}).json()
    c.get("/admin/api/keys", headers={"Authorization": f"Bearer {key['key']}"})
    c.get("/admin/api/members/whatever")
    c.get("/admin/nope")
    samples = [
        s for m in admin.app.state.metrics.registry.collect() if m.name == "decision_api_requests"
        for s in m.samples if s.name == "decision_api_requests_total"
    ]  # fmt: skip
    admin_samples = [s for s in samples if s.labels["route"] == "admin"]
    assert admin_samples and all(s.labels["key_id"] == "none" for s in admin_samples)
    assert not any(s.labels["route"].startswith("/admin") for s in samples)


def test_access_log_has_no_member_emails(admin, caplog):
    invite(admin, ALICE)
    owner = admin.signed_in(OWNER)
    caplog.set_level(logging.INFO, logger="decision_api.access")
    admin.call(owner, "DELETE", "/admin/api/members/alice@example.com")
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "decision_api.access"]
    assert any(e["path"] == "/admin/api/members/{email}" for e in lines)
    assert ALICE not in caplog.text


# ---------------------------------------------------------------- HTML pages and security headers


def _assert_security_headers(resp):
    csp = resp.headers["content-security-policy"]
    for part in ("default-src 'self'", "script-src 'self'", "frame-ancestors 'none'", "base-uri 'none'",
                 "form-action 'self'", "object-src 'none'"):  # fmt: skip
        assert part in csp, part
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert resp.headers["cache-control"] == "no-store"


def _assert_no_inline_code(html: str):
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline script"
    assert " style=" not in html and "<style" not in html, "inline style"
    assert not re.search(r"\son[a-z]+=", html), "inline event handler"
    assert not re.search(r"(src|href)=\"(https?:)?//", html), "external resource"


def test_security_headers_everywhere(admin):
    c = admin.signed_in(OWNER)
    for resp in (c.get("/admin"), c.get("/admin/members"), c.get("/admin/api/me"), admin.client().get("/admin"),
                 admin.client().get("/admin/auth/login"), admin.sign_in(admin.client(), MALLORY),
                 c.get("/admin/api/keys/x")):  # fmt: skip
        _assert_security_headers(resp)


def test_static_assets(admin):
    c = admin.client()
    js = c.get("/admin/static/admin.js")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    css = c.get("/admin/static/admin.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    for bad in ("/admin/static/../app.py", "/admin/static/%2e%2e%2fapp.py", "/admin/static/nope.js"):
        assert c.get(bad).status_code == 404, bad


def test_home_page_member(admin):
    invite(admin, ALICE)
    c = admin.signed_in(ALICE)
    created = admin.call(c, "POST", "/admin/api/keys", {"name": "<script>alert(1)</script>"}).json()
    resp = c.get("/admin")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    html = resp.text
    _assert_no_inline_code(html)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html and "<script>alert" not in html
    assert created["prefix"] in html and created["key"] not in html
    assert '<script src="/admin/static/admin.js" defer></script>' in html
    assert re.search(r'<meta name="csrf-token" content="[A-Za-z0-9_-]{32,}">', html)
    assert '<label for="key-name"' in html and '<label for="key-expires"' in html
    assert "you won't see it again" in html.lower()
    assert "Usage" in html and ALICE in html
    for owner_link in ('href="/admin/members"', 'href="/admin/audit"', 'href="/admin/keys"'):
        assert owner_link not in html


@pytest.mark.parametrize("path", ["/admin/keys", "/admin/usage", "/admin/members", "/admin/audit"])
def test_owner_pages_forbidden_for_members(admin, path):
    invite(admin, ALICE)
    resp = admin.signed_in(ALICE).get(path)
    assert resp.status_code == 403 and resp.headers["content-type"].startswith("text/html")


def test_owner_pages(admin):
    invite(admin, ALICE)
    alice = admin.signed_in(ALICE)
    admin.call(alice, "POST", "/admin/api/keys", {"name": "alice-key"})
    c = admin.signed_in(OWNER)
    home = c.get("/admin").text
    for link in ('href="/admin/members"', 'href="/admin/audit"', 'href="/admin/keys"', 'href="/admin/usage"'):
        assert link in home
    members = c.get("/admin/members")
    assert members.status_code == 200 and ALICE in members.text
    assert "test user" in members.text.lower() and "huynle-infra" in members.text
    assert '<label for="invite-email"' in members.text
    keys = c.get("/admin/keys", params={"user": ALICE})
    assert "alice-key" in keys.text and "imported (SOPS)" not in keys.text
    assert "imported (SOPS)" in c.get("/admin/keys").text
    assert c.get("/admin/keys", params={"user": "bad"}).status_code == 400
    assert c.get("/admin/usage").status_code == 200
    audit_page = c.get("/admin/audit")
    assert audit_page.status_code == 200 and "sign_in.allowed" in audit_page.text
    for resp in (members, keys, audit_page):
        _assert_no_inline_code(resp.text)
