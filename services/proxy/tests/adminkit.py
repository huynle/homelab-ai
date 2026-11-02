"""Helpers for the admin page tests: an app in SQLite mode with Google sign-in wired to :mod:`fakegoogle`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import create_app
from config import Settings
from conftest import API_KEY, IMAGE_URL, PEPPER, TEXT_URL, Clock
from fakegoogle import CLIENT_ID, CLIENT_SECRET, PUBLIC_URL, FakeGoogle

SESSION_KEY = b"s" * 32 + b"-session-key-not-a-secret"
ORIGIN = {"Origin": PUBLIC_URL}

# Every JSON route of api/admin-api.yaml (paths relative to the server, /admin), with its method.
API_ROUTES = [
    ("GET", "/admin/api/me"),
    ("GET", "/admin/api/keys"),
    ("POST", "/admin/api/keys"),
    ("PATCH", "/admin/api/keys/k7m2q4x6w3ab"),
    ("POST", "/admin/api/keys/k7m2q4x6w3ab/revoke"),
    ("GET", "/admin/api/usage"),
    ("GET", "/admin/api/members"),
    ("POST", "/admin/api/members"),
    ("DELETE", "/admin/api/members/alice@example.com"),
    ("GET", "/admin/api/audit"),
]
UNSAFE_ROUTES = [r for r in API_ROUTES if r[0] != "GET"] + [("POST", "/admin/auth/logout")]
BODIES = {
    ("POST", "/admin/api/keys"): {"name": "k"},
    ("PATCH", "/admin/api/keys/k7m2q4x6w3ab"): {"name": "k2"},
    ("POST", "/admin/api/members"): {"email": "carol@example.com"},
}


def write_secret(path: Path, data: bytes) -> str:
    path.write_bytes(data + b"\n")
    os.chmod(path, 0o600)
    return str(path)


def admin_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        api_keys=(API_KEY,),
        text_backend_url=TEXT_URL,
        image_backend_url=IMAGE_URL,
        keystore="sqlite",
        db_path=str(tmp_path / "decision-api.db"),
        pepper_file=write_secret(tmp_path / "pepper", PEPPER),
        owner_emails=("owner@example.com",),
        usage_flush_s=3600,
        oauth_client_id=CLIENT_ID,
        oauth_client_secret_file=write_secret(tmp_path / "oauth-secret", CLIENT_SECRET.encode()),
        public_url=PUBLIC_URL,
        session_key_file=write_secret(tmp_path / "session-key", SESSION_KEY),
    )
    base.update(overrides)
    return Settings(**base)


@dataclass
class Admin:
    """A running app plus the fake Google, the clock and a few request helpers."""

    app: FastAPI
    google: FakeGoogle
    clock: Clock
    clients: list[TestClient] = field(default_factory=list)

    @property
    def db(self):
        return self.app.state.db

    def client(self) -> TestClient:
        c = TestClient(self.app, base_url=PUBLIC_URL, follow_redirects=False, raise_server_exceptions=False)
        c.__enter__()
        self.clients.append(c)
        return c

    def start_login(self, c: TestClient, next_: str | None = None) -> dict:
        params = {"next": next_} if next_ is not None else {}
        resp = c.get("/admin/auth/login", params=params)
        assert resp.status_code == 302, resp.text
        return {k: v[0] for k, v in parse_qs(urlsplit(resp.headers["location"]).query).items()}

    def sign_in(self, c: TestClient, email: str, next_: str | None = None, **claims) -> httpx.Response:
        q = self.start_login(c, next_)
        claims.setdefault("nonce", q["nonce"])
        code = self.google.issue_code(q["code_challenge"], email=email, **claims)
        return c.get("/admin/auth/callback", params={"code": code, "state": q["state"]})

    def signed_in(self, email: str) -> TestClient:
        c = self.client()
        resp = self.sign_in(c, email)
        assert resp.status_code == 302, resp.text
        return c

    @staticmethod
    def csrf(c: TestClient) -> str:
        return c.get("/admin/api/me").json()["csrf_token"]

    def call(self, c: TestClient, method: str, url: str, json=None, *, csrf: str | None = "auto", **kw):
        headers = dict(ORIGIN)
        if csrf == "auto":
            csrf = self.csrf(c)
        if csrf is not None:
            headers["X-CSRF-Token"] = csrf
        headers.update(kw.pop("headers", {}))
        return c.request(method, url, json=json, headers=headers, **kw)

    def close(self):
        for c in self.clients:
            c.__exit__(None, None, None)
        self.app.state.db.close()


def make_admin(tmp_path: Path, backends, clock: Clock, **overrides) -> Admin:
    google = FakeGoogle(now=lambda: clock().timestamp())
    app = create_app(
        admin_settings(tmp_path, **overrides),
        transport=httpx.MockTransport(backends.handler),
        oauth_transport=google.transport,
        clock=clock,
    )
    return Admin(app=app, google=google, clock=clock)


def set_cookie_headers(resp: httpx.Response, name: str) -> list[str]:
    return [h for h in resp.headers.get_list("set-cookie") if h.startswith(name + "=")]
