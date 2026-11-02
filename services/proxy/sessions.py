"""Server-side admin sessions (the ``sessions`` table) and their CSRF tokens.

The browser holds only an opaque random id in the ``__Host-decision_session`` cookie; the table stores its
sha256 and the sha256 of the session's CSRF token, never the values. The CSRF token is derived from the id with
HMAC-SHA256 under the session key, so ``GET /admin/api/me`` can return it without storing it. Sessions last 12
hours from sign-in (not sliding). A session is valid only while its user is the owner or an active member, so
removing a member ends their sessions at once (``keyservice.remove_member`` also deletes them).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from db import Database, iso
from keyservice import Viewer, normalize_email

SESSION_COOKIE = "__Host-decision_session"
SESSION_TTL = timedelta(hours=12)
MAX_COOKIE_LEN = 128
_CSRF_CONTEXT = b"decision-api csrf v1\x00"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def csrf_token(session_key: bytes, session_id: str) -> str:
    return hmac.new(session_key, _CSRF_CONTEXT + session_id.encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Session:
    id_hash: str
    viewer: Viewer
    csrf_hash: str
    csrf_token: str
    expires_at: str

    def csrf_ok(self, presented: str | None) -> bool:
        """Constant-time check of an ``X-CSRF-Token`` value against the stored hash."""
        if not presented:
            return False
        return hmac.compare_digest(_sha256(presented), self.csrf_hash)


def create(db: Database, session_key: bytes, email: str, now: datetime) -> tuple[str, str]:
    """Start a session for ``email``; returns ``(session_id, csrf_token)``. Also prunes ended sessions."""
    session_id = secrets.token_urlsafe(32)
    token = csrf_token(session_key, session_id)
    now_iso = iso(now)
    with db.write(bump=False) as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ? OR revoked_at IS NOT NULL", (now_iso,))
        conn.execute(
            "INSERT INTO sessions (id_hash, email, csrf_token_hash, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (_sha256(session_id), normalize_email(email), _sha256(token), now_iso, iso(now + SESSION_TTL)),
        )
    return session_id, token


def lookup(db: Database, session_key: bytes, session_id: str | None, now: datetime) -> Session | None:
    """The live session for a cookie value, or None (unknown, expired, revoked, or the user was removed)."""
    if not session_id or len(session_id) > MAX_COOKIE_LEN:
        return None
    with db.read() as conn:
        row = conn.execute(
            "SELECT s.id_hash, s.csrf_token_hash, s.expires_at, u.email, u.role FROM sessions s"
            " JOIN users u ON u.email = s.email"
            " WHERE s.id_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.removed_at IS NULL",
            (_sha256(session_id), iso(now)),
        ).fetchone()
    if row is None:
        return None
    token = csrf_token(session_key, session_id)
    if not hmac.compare_digest(_sha256(token), row["csrf_token_hash"]):  # the session key was rotated
        return None
    return Session(
        id_hash=row["id_hash"],
        viewer=Viewer(email=row["email"], role=row["role"]),
        csrf_hash=row["csrf_token_hash"],
        csrf_token=token,
        expires_at=row["expires_at"],
    )


def revoke(db: Database, session: Session, now: datetime) -> None:
    with db.write(bump=False) as conn:
        conn.execute(
            "UPDATE sessions SET revoked_at = ? WHERE id_hash = ? AND revoked_at IS NULL", (iso(now), session.id_hash)
        )


def mark_sign_in(db: Database, email: str, now: datetime) -> None:
    with db.write(bump=False) as conn:
        conn.execute("UPDATE users SET last_sign_in_at = ? WHERE email = ?", (iso(now), normalize_email(email)))
