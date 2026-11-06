"""The admin page at ``/admin`` (api/admin-api.yaml): Google sign-in, sessions, CSRF, the JSON API and HTML pages.

Active only when ``KEYSTORE=sqlite`` and all four sign-in settings are set and valid (:func:`load_config`);
otherwise no admin route exists and the access middleware answers 404 for every ``/admin`` path.

Auth model:

- ``/admin/auth/login``, ``/admin/auth/callback``, ``/admin/static/*`` and the signed-out ``/admin`` page are
  public. Everything else needs the server-side session in ``__Host-decision_session``. Bearer API keys never
  authenticate ``/admin`` and the session cookie never authenticates ``/v1``.
- The JSON API never redirects: no session is 401 ``unauthorized``. A request whose ``Origin`` is present and not
  the public origin is 403 ``csrf_failed``; ``POST``/``PATCH``/``DELETE`` also need ``Origin`` and
  ``X-CSRF-Token`` (checked in constant time against the stored hash).
- Every ``/admin`` response gets :data:`SECURITY_HEADERS` and ``Cache-Control: no-store`` (see :func:`secure`).
"""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

import admin_pages as pages
import keyservice as ks
import oidc
import sessions
from config import Settings
from db import Database
from keyservice import ServiceError

log = logging.getLogger("decision_api")

PREFIX = "/admin"
CALLBACK_PATH = "/admin/auth/callback"
MIN_SESSION_KEY_BYTES = 32
MAX_BODY = 16 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Strict-Transport-Security": "max-age=31536000",
}

STATUS = {
    "invalid_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "csrf_failed": 403,
    "not_found": 404,
    "conflict": 409,
    "internal_error": 500,
}
CSRF_MESSAGE = "missing or invalid CSRF token or Origin"
OAUTH_SETTINGS = (
    ("oauth_client_id", "DECISION_API_OAUTH_CLIENT_ID"),
    ("oauth_client_secret_file", "DECISION_API_OAUTH_CLIENT_SECRET_FILE"),
    ("public_url", "DECISION_API_PUBLIC_URL"),
    ("session_key_file", "DECISION_API_SESSION_KEY_FILE"),
)


def is_admin_path(path: str) -> bool:
    return path == PREFIX or path.startswith(PREFIX + "/")


def log_path(path: str) -> str:
    """The path for the access log: member emails in ``/admin/api/members/{email}`` are not logged."""
    if path.startswith("/admin/api/members/"):
        return "/admin/api/members/{email}"
    return path


class AdminError(Exception):
    def __init__(self, code: str, message: str):
        assert code in STATUS, code
        super().__init__(message)
        self.code = code
        self.message = message


def error_json(code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=STATUS[code])


def secure(response: Response) -> Response:
    for name, value in SECURITY_HEADERS.items():
        response.headers[name] = value
    if "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------- configuration


@dataclass(frozen=True)
class AdminConfig:
    client_id: str
    public_url: str  # the origin, e.g. https://decide.ai.huynle.com
    client_secret: str = field(repr=False)
    session_key: bytes = field(repr=False)

    @property
    def redirect_uri(self) -> str:
        return self.public_url + CALLBACK_PATH


def _read_secret(path: str, var: str) -> bytes:
    try:
        return Path(path).read_bytes().strip()
    except OSError as exc:
        raise RuntimeError(f"{var} {path!r} is not readable ({exc.strerror})") from None


def _origin(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise RuntimeError(
            "DECISION_API_PUBLIC_URL must be an https origin without path, query or user info "
            f"(for example https://decide.ai.huynle.com), got {url!r}"
        )
    return f"https://{parts.netloc.lower()}"


def load_config(settings: Settings, pepper: bytes | None) -> AdminConfig | None:
    """The admin sign-in config, None when the admin page is off, RuntimeError when it is half-configured."""
    present = [var for attr, var in OAUTH_SETTINGS if getattr(settings, attr)]
    if settings.keystore != "sqlite":
        if present:
            log.warning("admin page disabled: it needs KEYSTORE=sqlite (ignoring %s)", ", ".join(present))
        return None
    if not present:
        log.info("admin page disabled: DECISION_API_OAUTH_* / DECISION_API_PUBLIC_URL / SESSION_KEY_FILE not set")
        return None
    missing = [var for attr, var in OAUTH_SETTINGS if not getattr(settings, attr)]
    if missing:
        raise RuntimeError(f"admin sign-in is partly configured; also set {', '.join(missing)}")
    client_id = settings.oauth_client_id.strip()
    if not client_id or any(c.isspace() for c in client_id):
        raise RuntimeError("DECISION_API_OAUTH_CLIENT_ID is not a valid client id")
    secret = _read_secret(settings.oauth_client_secret_file, "DECISION_API_OAUTH_CLIENT_SECRET_FILE")
    if not secret:
        raise RuntimeError("DECISION_API_OAUTH_CLIENT_SECRET_FILE is empty")
    session_key = _read_secret(settings.session_key_file, "DECISION_API_SESSION_KEY_FILE")
    if len(session_key) < MIN_SESSION_KEY_BYTES:
        raise RuntimeError(f"DECISION_API_SESSION_KEY_FILE must hold at least {MIN_SESSION_KEY_BYTES} bytes")
    if pepper is not None and hmac.compare_digest(session_key, pepper):
        raise RuntimeError("DECISION_API_SESSION_KEY_FILE must differ from DECISION_API_KEY_PEPPER_FILE")
    try:
        secret_text = secret.decode()
    except UnicodeDecodeError:
        raise RuntimeError("DECISION_API_OAUTH_CLIENT_SECRET_FILE is not UTF-8 text") from None
    return AdminConfig(
        client_id=client_id,
        public_url=_origin(settings.public_url),
        client_secret=secret_text,
        session_key=session_key,
    )


# ---------------------------------------------------------------- the admin app


def _load_static() -> dict[str, tuple[bytes, str]]:
    return {p.name: (p.read_bytes(), STATIC_TYPES[p.suffix]) for p in STATIC_DIR.iterdir() if p.suffix in STATIC_TYPES}


class Admin:
    def __init__(self, config: AdminConfig, db: Database, google: oidc.GoogleOIDC, now: Callable[[], datetime]):
        self.config = config
        self.db = db
        self.google = google
        self.now = now
        self.static = _load_static()

    def __repr__(self) -> str:
        return f"Admin(client_id={self.config.client_id!r}, public_url={self.config.public_url!r})"

    # ---- auth helpers

    def session(self, request: Request) -> sessions.Session | None:
        value = request.cookies.get(sessions.SESSION_COOKIE)
        return sessions.lookup(self.db, self.config.session_key, value, self.now())

    def api_session(self, request: Request, unsafe: bool) -> sessions.Session:
        s = self.session(request)
        if s is None:
            raise AdminError("unauthorized", "sign in required")
        origin = request.headers.get("origin")
        if origin is not None and origin != self.config.public_url:
            raise AdminError("csrf_failed", CSRF_MESSAGE)
        if unsafe and (origin is None or not s.csrf_ok(request.headers.get("x-csrf-token"))):
            raise AdminError("csrf_failed", CSRF_MESSAGE)
        return s

    def _cookie(self, response: Response, name: str, value: str, max_age: int) -> None:
        response.set_cookie(name, value, max_age=max_age, path="/", secure=True, httponly=True, samesite="lax")

    def _clear(self, response: Response, name: str) -> None:
        response.delete_cookie(name, path="/", secure=True, httponly=True, samesite="lax")

    # ---- sign-in

    def deny(self, reason: str, actor: str | None = None) -> None:
        ks.record_audit(self.db, "sign_in.denied", actor_email=actor, detail={"reason": reason}, now=self.now())

    async def callback(self, request: Request) -> Response:
        q = request.query_params
        now = self.now()
        login = oidc.read_login(self.config.session_key, request.cookies.get(oidc.LOGIN_COOKIE), now.timestamp())
        state = q.get("state") or ""

        def done(resp: Response) -> Response:
            self._clear(resp, oidc.LOGIN_COOKIE)  # the login cookie is single-use
            return resp

        def page(status: int, title: str, text: str) -> Response:
            link = ("/admin/auth/login", "Try again") if status == 400 else None
            return done(HTMLResponse(pages.message(title, text, link), status_code=status))

        not_allowed = "This Google account is not allowed. Ask the owner for an invite."
        if login is None or len(state) > 256 or not hmac.compare_digest(state.encode(), login.state.encode()):
            self.deny("state_mismatch")
            return page(400, "Sign-in failed", "Sign-in expired or was not started here.")
        if q.get("error"):
            self.deny("google_error")
            return page(403, "Not allowed", not_allowed)
        code = q.get("code")
        if not code or len(code) > 2048:
            self.deny("google_error")
            return page(400, "Sign-in failed", "Google did not return a sign-in code.")
        try:
            token = await self.google.exchange(code, login.verifier)
            claims = await self.google.verify_id_token(token, login.nonce)
        except oidc.OidcError as exc:
            log.warning("admin sign-in failed: %s", exc)
            self.deny("token_invalid")
            return page(400, "Sign-in failed", "Google sign-in could not be verified.")
        email = ks.normalize_email(claims["email"])
        if claims.get("email_verified") is not True:
            self.deny("email_unverified", email)
            return page(403, "Not allowed", not_allowed)
        viewer = ks.get_viewer(self.db, email)
        if viewer is None:
            self.deny("not_invited", email)
            return page(403, "Not allowed", not_allowed)
        now = self.now()
        session_id, _ = sessions.create(self.db, self.config.session_key, viewer.email, now)
        sessions.mark_sign_in(self.db, viewer.email, now)
        ks.record_audit(self.db, "sign_in.allowed", actor_email=viewer.email, detail={"role": viewer.role}, now=now)
        resp = RedirectResponse(login.next, status_code=302)
        self._cookie(resp, sessions.SESSION_COOKIE, session_id, int(sessions.SESSION_TTL.total_seconds()))
        return done(resp)

    # ---- request bodies and query strings

    async def body(self, request: Request, allowed: set[str], required: bool = True) -> dict:
        declared = request.headers.get("content-length")
        too_large = AdminError("invalid_request", f"request body exceeds {MAX_BODY} bytes")
        if declared and declared.isdigit() and int(declared) > MAX_BODY:
            raise too_large
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_BODY:
                raise too_large
        if not buf.strip() and not required:
            return {}
        try:
            data = json.loads(buf)
        except (ValueError, UnicodeDecodeError):
            raise AdminError("invalid_request", "request body is not valid JSON") from None
        if not isinstance(data, dict):
            raise AdminError("invalid_request", "request body must be a JSON object")
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise AdminError("invalid_request", f"{unknown[0]}: unknown field")
        return data

    @staticmethod
    def int_param(request: Request, name: str, default: int | None, message: str) -> int | None:
        value = request.query_params.get(name)
        if value is None:
            return default
        if not value.isdigit() or len(value) > 12:
            raise AdminError("invalid_request", message)
        return int(value)

    # ---- pages

    def page_viewer(self, request: Request, owner_only: bool) -> tuple[sessions.Session | None, Response | None]:
        s = self.session(request)
        if s is None:
            return None, RedirectResponse(f"/admin/auth/login?{urlencode({'next': request.url.path})}", 302)
        if owner_only and not s.viewer.is_owner:
            return s, HTMLResponse(pages.message("Forbidden", "This page is for the owner only."), status_code=403)
        return s, None


def service_error_page(exc: ServiceError) -> HTMLResponse:
    return HTMLResponse(pages.message("Request failed", exc.message, ("/admin", "Back")), status_code=exc.status)


def install(app: FastAPI, admin: Admin) -> None:
    """Register the admin routes and their error handlers on ``app``."""

    @app.exception_handler(AdminError)
    async def _admin_error(request: Request, exc: AdminError):
        return error_json(exc.code, exc.message)

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        return error_json(exc.code, exc.message)

    def now() -> datetime:
        return admin.now()

    def today():
        return admin.now().date()

    # ---------------- auth

    @app.get("/admin/auth/login")
    async def login(request: Request):
        state = oidc.new_login(request.query_params.get("next"))
        resp = RedirectResponse(admin.google.authorization_url(state), status_code=302)
        value = oidc.sign_login(admin.config.session_key, state, now().timestamp())
        admin._cookie(resp, oidc.LOGIN_COOKIE, value, oidc.LOGIN_TTL_S)
        return resp

    @app.get("/admin/auth/callback")
    async def callback(request: Request):
        return await admin.callback(request)

    @app.post("/admin/auth/logout")
    async def logout(request: Request):
        s = admin.api_session(request, unsafe=True)
        sessions.revoke(admin.db, s, now())
        resp = Response(status_code=204)
        admin._clear(resp, sessions.SESSION_COOKIE)
        return resp

    # ---------------- JSON API

    @app.get("/admin/api/me")
    async def me(request: Request):
        s = admin.api_session(request, unsafe=False)
        return {"email": s.viewer.email, "role": s.viewer.role, "csrf_token": s.csrf_token}

    @app.get("/admin/api/keys")
    async def list_keys(request: Request):
        v = admin.api_session(request, unsafe=False).viewer
        return {"keys": ks.list_keys(admin.db, v, request.query_params.get("user"), now=now())}

    @app.post("/admin/api/keys")
    async def create_key(request: Request):
        v = admin.api_session(request, unsafe=True).viewer
        data = await admin.body(request, {"name", "expires_at"})
        created = ks.create_key(admin.db, v, data.get("name"), data.get("expires_at"), now=now())
        return JSONResponse(created, status_code=201, headers={"Location": f"/admin/api/keys/{created['id']}"})

    @app.patch("/admin/api/keys/{id}")
    async def update_key(request: Request, id: str):
        v = admin.api_session(request, unsafe=True).viewer
        data = await admin.body(request, {"name", "allow_paid"})
        if "name" in data and data["name"] is None:
            raise AdminError("invalid_request", f"name: must be 1 to {ks.MAX_NAME} characters")
        if "allow_paid" in data and not v.is_owner:
            raise AdminError("forbidden", "only the owner can change paid model access")
        if "allow_paid" in data and not isinstance(data["allow_paid"], bool):
            raise AdminError("invalid_request", "allow_paid: must be true or false")
        key = ks.rename_key(admin.db, v, id, data.get("name"), now=now())
        if "allow_paid" in data:
            key = ks.set_paid_access(admin.db, v, id, data["allow_paid"], now=now())
        return key

    @app.post("/admin/api/keys/{id}/revoke")
    async def revoke_key(request: Request, id: str):
        v = admin.api_session(request, unsafe=True).viewer
        await admin.body(request, set(), required=False)
        return ks.revoke_key(admin.db, v, id, now=now())

    @app.get("/admin/api/usage")
    async def usage(request: Request):
        v = admin.api_session(request, unsafe=False).viewer
        q = request.query_params
        return ks.usage_query(
            admin.db, v, q.get("from"), q.get("to"), q.get("key_id"), q.get("user"), q.get("group_by"), today=today()
        )

    @app.get("/admin/api/members")
    async def list_members(request: Request):
        v = admin.api_session(request, unsafe=False).viewer
        return {"members": ks.list_members(admin.db, v, now=now())}

    @app.post("/admin/api/members")
    async def invite_member(request: Request):
        v = admin.api_session(request, unsafe=True).viewer
        if not v.is_owner:
            raise AdminError("forbidden", "owner only")
        data = await admin.body(request, {"email", "role"})
        if data.get("role", ks.MEMBER) != ks.MEMBER:
            raise AdminError("invalid_request", "role: must be member")
        member = ks.invite_member(admin.db, v, data.get("email"), now=now())
        location = f"/admin/api/members/{quote(member['email'], safe='@')}"
        return JSONResponse(member, status_code=201, headers={"Location": location})

    @app.delete("/admin/api/members/{email}")
    async def remove_member(request: Request, email: str):
        v = admin.api_session(request, unsafe=True).viewer
        return ks.remove_member(admin.db, v, email, now=now())

    @app.get("/admin/api/audit")
    async def audit(request: Request):
        v = admin.api_session(request, unsafe=False).viewer
        limit = admin.int_param(request, "limit", 100, f"limit: must be an integer from 1 to {ks.MAX_AUDIT_LIMIT}")
        before = admin.int_param(request, "before", None, "before: must be a positive integer")
        q = request.query_params
        return ks.audit_query(admin.db, v, q.get("from"), q.get("to"), before, limit, today=today())

    # ---------------- HTML pages

    def html(body: str) -> HTMLResponse:
        return HTMLResponse(body)

    @app.get("/admin")
    async def home(request: Request):
        s = admin.session(request)
        if s is None:
            return html(pages.signed_out())
        v = s.viewer
        keys = ks.list_keys(admin.db, v, v.email, now=now())
        report = ks.usage_query(admin.db, v, user=v.email, group_by="day", today=today())
        return html(pages.home(v, s.csrf_token, keys, report))

    def owner_page(render: Callable[[Request, sessions.Session], str]):
        async def handler(request: Request):
            s, early = admin.page_viewer(request, owner_only=True)
            if early is not None:
                return early
            try:
                return html(render(request, s))
            except ServiceError as exc:
                return service_error_page(exc)

        return handler

    def members(request: Request, s: sessions.Session) -> list[dict]:
        return ks.list_members(admin.db, s.viewer, now=now())

    def keys_page(request: Request, s: sessions.Session) -> str:
        user = request.query_params.get("user") or None
        keys = ks.list_keys(admin.db, s.viewer, user, now=now())
        return pages.all_keys(s.viewer, s.csrf_token, keys, members(request, s), ks.normalize_email(user) or None)

    def usage_page(request: Request, s: sessions.Session) -> str:
        q = request.query_params
        user = q.get("user") or None
        args: dict[str, Any] = {"from_": q.get("from") or None, "to": q.get("to") or None, "user": user}
        by_day = ks.usage_query(admin.db, s.viewer, group_by="day", today=today(), **args)
        by_key = ks.usage_query(admin.db, s.viewer, group_by="key", today=today(), **args)
        selected = ks.normalize_email(user) or None
        return pages.usage_page(s.viewer, s.csrf_token, by_day, by_key, members(request, s), selected)

    def members_page(request: Request, s: sessions.Session) -> str:
        return pages.members_page(s.viewer, s.csrf_token, members(request, s))

    def audit_page(request: Request, s: sessions.Session) -> str:
        before = request.query_params.get("before")
        if before is not None and not before.isdigit():
            raise ServiceError("invalid_request", "before: must be a positive integer")
        page = ks.audit_query(admin.db, s.viewer, before=int(before) if before else None, limit=50, today=today())
        return pages.audit_page(s.viewer, s.csrf_token, page, {})

    app.add_api_route("/admin/keys", owner_page(keys_page), methods=["GET"])
    app.add_api_route("/admin/usage", owner_page(usage_page), methods=["GET"])
    app.add_api_route("/admin/members", owner_page(members_page), methods=["GET"])
    app.add_api_route("/admin/audit", owner_page(audit_page), methods=["GET"])

    @app.get("/admin/static/{name}")
    async def static(name: str):
        found = admin.static.get(name)
        if found is None:
            return error_json("not_found", "not found")
        content, media_type = found
        return Response(content, media_type=media_type, headers={"Cache-Control": "no-cache"})
