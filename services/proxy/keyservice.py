"""Key service layer for the admin API (api/admin-api.yaml): keys, members, usage and audit.

Plain functions over a :class:`db.Database`; the HTTP routes and Google sign-in live elsewhere. Each function
takes the signed-in :class:`Viewer` and enforces the owner/member rules of the contract, raising
:class:`ServiceError` with an admin error code. Results are JSON-ready dicts in the contract's shapes.
Pass ``now=`` (an aware UTC datetime) or ``today=`` to pin the clock in tests.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from db import UTC, Database, iso, utcnow
from keystore import SOURCE_IMPORTED, SOURCE_ISSUED, generate_key, hash_key, new_public_id

log = logging.getLogger("decision_api")

OWNER = "owner"
MEMBER = "member"
MAX_MEMBER_KEYS = 20
MAX_NAME = 64
MAX_EXPIRY = timedelta(days=730)
MAX_EMAIL = 254
DEFAULT_USAGE_DAYS = 30
MAX_RANGE_DAYS = 366
MAX_AUDIT_LIMIT = 500
IMPORTED_NAME = "imported (SOPS)"
IMPORTED_PREFIX = "imported…"
GROUP_BY = ("day", "key", "model", "route")
AUDIT_EVENTS = frozenset(
    {
        "sign_in.allowed",
        "sign_in.denied",
        "member.invited",
        "member.removed",
        "key.created",
        "key.revoked",
        "key.paid_access_changed",
    }
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
KEY_ID_RE = re.compile(r"^[a-z2-7]{12}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

STATUS = {"invalid_request": 400, "forbidden": 403, "not_found": 404, "conflict": 409}


class ServiceError(Exception):
    """A request the rules refuse. ``code`` is an admin API error code; ``status`` its HTTP status."""

    def __init__(self, code: str, message: str):
        assert code in STATUS, code
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def status(self) -> int:
        return STATUS[self.code]


@dataclass(frozen=True)
class Viewer:
    email: str
    role: str

    @property
    def is_owner(self) -> bool:
        return self.role == OWNER


# ---------------------------------------------------------------- helpers


def _now(now: datetime | None) -> datetime:
    now = utcnow() if now is None else now
    if now.tzinfo is None:
        raise ValueError("naive datetime; pass an aware (UTC) datetime")
    return now


def normalize_email(email: Any) -> str:
    return email.strip().lower() if isinstance(email, str) else ""


def _valid_email(email: Any, field: str = "email") -> str:
    e = normalize_email(email)
    if len(e) > MAX_EMAIL or not EMAIL_RE.fullmatch(e):
        raise ServiceError("invalid_request", f"{field}: must be a valid email address")
    return e


def _name(name: Any) -> str:
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= MAX_NAME:
        raise ServiceError("invalid_request", f"name: must be 1 to {MAX_NAME} characters")
    return name.strip()


def _expiry(expires_at: datetime | str | None, now: datetime) -> str | None:
    if expires_at is None:
        return None
    if isinstance(expires_at, str):
        try:
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))  # noqa: FURB162 (3.10)
        except ValueError:
            raise ServiceError("invalid_request", "expires_at: must be an RFC 3339 date-time") from None
    if not isinstance(expires_at, datetime) or expires_at.tzinfo is None:
        raise ServiceError("invalid_request", "expires_at: must be a date-time with a time zone")
    if expires_at <= now:
        raise ServiceError("invalid_request", "expires_at: must be in the future")
    if expires_at > now + MAX_EXPIRY:
        raise ServiceError("invalid_request", "expires_at: must be at most 2 years from now")
    return iso(expires_at)


def _status(row: Mapping, now_iso: str) -> str:
    if row["revoked_at"]:
        return "revoked"
    if row["expires_at"] and row["expires_at"] <= now_iso:
        return "expired"
    return "active"


def _key_dict(row: Mapping, now_iso: str) -> dict:
    return {
        "id": row["public_id"],
        "name": row["name"],
        "prefix": row["prefix"],
        "source": row["source"],
        "owner_email": row["owner_email"],
        "status": _status(row, now_iso),
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
        "revoked_at": row["revoked_at"],
        "last_used_at": row["last_used_at"],
        "rate_limit_rpm": row["rate_limit_rpm"],
        "allow_paid": bool(row["allow_paid"]),
    }


def _pepper(db: Database) -> bytes:
    if not db.pepper:
        raise RuntimeError("the database was opened without a key pepper")
    return db.pepper


ACTIVE_KEY_SQL = "revoked_at IS NULL AND (expires_at IS NULL OR expires_at > ?)"


def _require_owner(viewer: Viewer) -> None:
    if not viewer.is_owner:
        raise ServiceError("forbidden", "owner only")


def _user_filter(viewer: Viewer, user: Any) -> str | None:
    """The email whose keys to show: a member is limited to their own; the owner may pick anyone or all."""
    if user is None:
        return None if viewer.is_owner else viewer.email
    email = _valid_email(user, "user")
    if not viewer.is_owner and email != viewer.email:
        raise ServiceError("forbidden", "members can only see their own keys")
    return email


def _visible_key(conn: sqlite3.Connection, viewer: Viewer, public_id: Any) -> sqlite3.Row:
    """The key row if the viewer may see it. Another user's key looks the same as a missing one."""
    row = None
    if isinstance(public_id, str) and KEY_ID_RE.fullmatch(public_id):
        row = conn.execute("SELECT * FROM api_keys WHERE public_id = ?", (public_id,)).fetchone()
    if row is None or (not viewer.is_owner and row["owner_email"] != viewer.email):
        raise ServiceError("not_found", "key not found")
    return row


def _audit(
    conn: sqlite3.Connection,
    event: str,
    at: str,
    actor_email: str | None = None,
    target_email: str | None = None,
    key_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    if event not in AUDIT_EVENTS:
        raise ValueError(f"unknown audit event {event!r}")
    conn.execute(
        "INSERT INTO audit (at, actor_email, event, target_email, key_id, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (at, actor_email, event, target_email, key_id, json.dumps(dict(detail or {}))),
    )


def record_audit(
    db: Database,
    event: str,
    *,
    actor_email: str | None = None,
    target_email: str | None = None,
    key_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> None:
    """Append one audit entry (for example ``sign_in.allowed``). Never pass IPs, tokens or keys."""
    at = iso(_now(now))
    with db.write(bump=False) as conn:
        _audit(conn, event, at, actor_email, target_email, key_id, detail)


# ---------------------------------------------------------------- users


def ensure_owners(db: Database, emails: Iterable[str], *, now: datetime | None = None) -> None:
    """Make the configured owner emails owners. Owners no longer in config are demoted to members."""
    owners = sorted({normalize_email(e) for e in emails if normalize_email(e)})
    _now(now)
    with db.write() as conn:
        for email in owners:
            conn.execute(
                "INSERT INTO users (email, role) VALUES (?, 'owner')"
                " ON CONFLICT (email) DO UPDATE SET role = 'owner', removed_at = NULL",
                (email,),
            )
        marks = ",".join("?" * len(owners))
        demoted = conn.execute(
            f"UPDATE users SET role = 'member' WHERE role = 'owner' AND email NOT IN ({marks}) RETURNING email",
            owners,
        ).fetchall()
    for row in demoted:
        log.warning("%s is no longer in DECISION_API_OWNER_EMAILS: demoted to member", row["email"])


def get_viewer(db: Database, email: Any) -> Viewer | None:
    """The owner or an invited (not removed) member with this email, else None."""
    with db.read() as conn:
        row = conn.execute(
            "SELECT email, role FROM users WHERE email = ? AND removed_at IS NULL", (normalize_email(email),)
        ).fetchone()
    return Viewer(email=row["email"], role=row["role"]) if row else None


def _member_dict(conn: sqlite3.Connection, row: Mapping, now_iso: str) -> dict:
    active = conn.execute(
        f"SELECT count(*) FROM api_keys WHERE owner_email = ? AND {ACTIVE_KEY_SQL}", (row["email"], now_iso)
    ).fetchone()[0]
    return {
        "email": row["email"],
        "role": row["role"],
        "invited_at": row["invited_at"],
        "invited_by": row["invited_by"],
        "last_sign_in_at": row["last_sign_in_at"],
        "active_keys": active,
    }


def list_members(db: Database, viewer: Viewer, *, now: datetime | None = None) -> list[dict]:
    _require_owner(viewer)
    now_iso = iso(_now(now))
    with db.read() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE removed_at IS NULL"
            " ORDER BY role = 'owner' DESC, invited_at IS NOT NULL, invited_at, email"
        ).fetchall()
        return [_member_dict(conn, r, now_iso) for r in rows]


def invite_member(db: Database, viewer: Viewer, email: Any, *, now: datetime | None = None) -> dict:
    _require_owner(viewer)
    email = _valid_email(email)
    now_iso = iso(_now(now))
    with db.write() as conn:
        row = conn.execute("SELECT role, removed_at FROM users WHERE email = ?", (email,)).fetchone()
        if row is not None and row["role"] == OWNER:
            raise ServiceError("conflict", f"{email} is the owner")
        if row is not None and row["removed_at"] is None:
            raise ServiceError("conflict", f"{email} is already a member")
        conn.execute(
            "INSERT INTO users (email, role, invited_by, invited_at) VALUES (?, 'member', ?, ?)"
            " ON CONFLICT (email) DO UPDATE SET"
            "  role = 'member', invited_by = excluded.invited_by, invited_at = excluded.invited_at, removed_at = NULL",
            (email, viewer.email, now_iso),
        )
        _audit(conn, "member.invited", now_iso, viewer.email, email, detail={"role": MEMBER})
        return _member_dict(conn, conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone(), now_iso)


def remove_member(db: Database, viewer: Viewer, email: Any, *, now: datetime | None = None) -> dict:
    """In one transaction: remove from the allow list, revoke the member's active keys, delete their sessions."""
    _require_owner(viewer)
    email = normalize_email(email)
    now_iso = iso(_now(now))
    with db.write() as conn:
        row = conn.execute("SELECT role FROM users WHERE email = ? AND removed_at IS NULL", (email,)).fetchone()
        if row is None:
            raise ServiceError("not_found", "member not found")
        if row["role"] == OWNER:
            raise ServiceError("forbidden", "the owner cannot be removed")
        revoked = [
            r["public_id"]
            for r in conn.execute(
                f"SELECT public_id FROM api_keys WHERE owner_email = ? AND {ACTIVE_KEY_SQL} ORDER BY created_at, rowid",
                (email, now_iso),
            )
        ]
        for public_id in revoked:
            conn.execute("UPDATE api_keys SET revoked_at = ? WHERE public_id = ?", (now_iso, public_id))
            _audit(conn, "key.revoked", now_iso, viewer.email, email, public_id, {"reason": "member_removed"})
        conn.execute("DELETE FROM sessions WHERE email = ?", (email,))
        conn.execute("UPDATE users SET removed_at = ? WHERE email = ?", (now_iso, email))
        _audit(conn, "member.removed", now_iso, viewer.email, email, detail={"revoked_keys": len(revoked)})
    return {"email": email, "revoked_keys": revoked}


# ---------------------------------------------------------------- keys


def create_key(
    db: Database,
    viewer: Viewer,
    name: Any,
    expires_at: datetime | str | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """Issue a key for the viewer. The result's ``key`` is the full key: it is returned only here."""
    now = _now(now)
    now_iso = iso(now)
    name = _name(name)
    expires = _expiry(expires_at, now)
    with db.write() as conn:
        if not viewer.is_owner:
            active = conn.execute(
                f"SELECT count(*) FROM api_keys WHERE owner_email = ? AND {ACTIVE_KEY_SQL}", (viewer.email, now_iso)
            ).fetchone()[0]
            if active >= MAX_MEMBER_KEYS:
                raise ServiceError("conflict", f"at most {MAX_MEMBER_KEYS} active keys per member; revoke one first")
        for attempt in range(3):
            raw, public_id = generate_key()
            try:
                conn.execute(
                    "INSERT INTO api_keys (public_id, key_hmac, name, owner_email, source, prefix, created_at,"
                    " expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (public_id, hash_key(_pepper(db), raw), name, viewer.email, SOURCE_ISSUED,
                     f"dk_{public_id}_…", now_iso, expires),
                )  # fmt: skip
                break
            except sqlite3.IntegrityError:  # public id collision (60 random bits): try another
                if attempt == 2:
                    raise
        _audit(conn, "key.created", now_iso, viewer.email, viewer.email, public_id, {"name": name})
        row = conn.execute("SELECT * FROM api_keys WHERE public_id = ?", (public_id,)).fetchone()
    return {"key": raw, **_key_dict(row, now_iso)}


def list_keys(db: Database, viewer: Viewer, user: Any = None, *, now: datetime | None = None) -> list[dict]:
    """Members get their own keys; the owner every key or one user's. Newest first; revoked/expired included."""
    email = _user_filter(viewer, user)
    now_iso = iso(_now(now))
    sql, args = "SELECT * FROM api_keys", []
    if email is not None:
        sql, args = sql + " WHERE owner_email = ?", [email]
    with db.read() as conn:
        rows = conn.execute(sql + " ORDER BY created_at DESC, rowid DESC", args).fetchall()
    return [_key_dict(r, now_iso) for r in rows]


def get_key(db: Database, viewer: Viewer, public_id: Any, *, now: datetime | None = None) -> dict:
    now_iso = iso(_now(now))
    with db.read() as conn:
        return _key_dict(_visible_key(conn, viewer, public_id), now_iso)


def rename_key(db: Database, viewer: Viewer, public_id: Any, name: Any, *, now: datetime | None = None) -> dict:
    """Rename a key; ``name`` None is a no-op. A revoked key cannot be renamed."""
    now_iso = iso(_now(now))
    with db.write() as conn:
        row = _visible_key(conn, viewer, public_id)
        if name is not None:
            name = _name(name)
            if row["revoked_at"]:
                raise ServiceError("conflict", "key is revoked")
            conn.execute("UPDATE api_keys SET name = ? WHERE public_id = ?", (name, row["public_id"]))
            row = conn.execute("SELECT * FROM api_keys WHERE public_id = ?", (row["public_id"],)).fetchone()
    return _key_dict(row, now_iso)


def set_paid_access(db: Database, viewer: Viewer, public_id: Any, allow: Any, *, now: datetime | None = None) -> dict:
    """Owner only: let a key use paid (OpenRouter) models, or stop it. Applies on the key's next request.

    Audited as ``key.paid_access_changed`` when the value changes. A revoked key cannot be changed.
    """
    _require_owner(viewer)
    if not isinstance(allow, bool):
        raise ServiceError("invalid_request", "allow_paid: must be true or false")
    now_iso = iso(_now(now))
    with db.write() as conn:
        row = _visible_key(conn, viewer, public_id)
        if row["revoked_at"]:
            raise ServiceError("conflict", "key is revoked")
        if bool(row["allow_paid"]) != allow:
            conn.execute("UPDATE api_keys SET allow_paid = ? WHERE public_id = ?", (int(allow), row["public_id"]))
            _audit(
                conn,
                "key.paid_access_changed",
                now_iso,
                viewer.email,
                row["owner_email"],
                row["public_id"],
                {"allow_paid": allow},
            )
            row = conn.execute("SELECT * FROM api_keys WHERE public_id = ?", (row["public_id"],)).fetchone()
    return _key_dict(row, now_iso)


def revoke_key(db: Database, viewer: Viewer, public_id: Any, *, now: datetime | None = None) -> dict:
    """Soft, immediate, permanent. Idempotent: an already revoked key keeps its time and gets no new audit entry."""
    now_iso = iso(_now(now))
    with db.write() as conn:
        row = _visible_key(conn, viewer, public_id)
        if not row["revoked_at"]:
            conn.execute("UPDATE api_keys SET revoked_at = ? WHERE public_id = ?", (now_iso, row["public_id"]))
            _audit(conn, "key.revoked", now_iso, viewer.email, row["owner_email"], row["public_id"], {"reason": "user"})
            row = conn.execute("SELECT * FROM api_keys WHERE public_id = ?", (row["public_id"],)).fetchone()
    return _key_dict(row, now_iso)


def import_legacy_keys(
    db: Database, pepper: bytes, keys: Iterable[str], owner_email: str, *, now: datetime | None = None
) -> list[str]:
    """Import file keys as owner keys (``source: imported``), stored only as HMAC. Idempotent.

    A key whose HMAC is already stored (even revoked) is skipped, so a revoked imported key stays revoked.
    Returns the public ids of the keys imported now.
    """
    now_iso = iso(_now(now))
    owner = normalize_email(owner_email)
    imported = []
    with db.write() as conn:
        count = conn.execute("SELECT count(*) FROM api_keys WHERE source = ?", (SOURCE_IMPORTED,)).fetchone()[0]
        for raw in keys:
            digest = hash_key(pepper, raw)
            if conn.execute("SELECT 1 FROM api_keys WHERE key_hmac = ?", (digest,)).fetchone():
                continue
            count += 1
            name = IMPORTED_NAME if count == 1 else f"{IMPORTED_NAME} {count}"
            public_id = new_public_id()
            conn.execute(
                "INSERT INTO api_keys (public_id, key_hmac, name, owner_email, source, prefix, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (public_id, digest, name, owner, SOURCE_IMPORTED, IMPORTED_PREFIX, now_iso),
            )
            _audit(conn, "key.created", now_iso, None, owner, public_id, {"name": name, "source": SOURCE_IMPORTED})
            imported.append(public_id)
    return imported


# ---------------------------------------------------------------- usage and audit queries


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        raise ServiceError("invalid_request", f"{field}: must be a date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ServiceError("invalid_request", f"{field}: must be a date (YYYY-MM-DD)") from None


def _date_range(from_: Any, to: Any, today: date | None) -> tuple[date, date]:
    """``to`` defaults to today (UTC), ``from`` to 29 days before ``to``; both inclusive, at most 366 days."""
    today = utcnow().astimezone(UTC).date() if today is None else today
    end = today if to is None else _parse_date(to, "to")
    start = end - timedelta(days=DEFAULT_USAGE_DAYS - 1) if from_ is None else _parse_date(from_, "from")
    if start > end:
        raise ServiceError("invalid_request", "from: must not be after to")
    if (end - start).days + 1 > MAX_RANGE_DAYS:
        raise ServiceError("invalid_request", f"from: the range is at most {MAX_RANGE_DAYS} days")
    return start, end


def _totals(requests: int, client_errors: int, server_errors: int, latency: float, cost: float = 0.0) -> dict:
    return {
        "requests": requests,
        "errors": client_errors + server_errors,
        "client_errors": client_errors,
        "server_errors": server_errors,
        "total_latency_ms": round(float(latency), 3),
        "avg_latency_ms": round(latency / requests, 4) if requests else None,
        "cost_usd": round(float(cost or 0.0), 9),
    }


def usage_query(
    db: Database,
    viewer: Viewer,
    from_: Any = None,
    to: Any = None,
    key_id: Any = None,
    user: Any = None,
    group_by: Any = "day",
    *,
    today: date | None = None,
) -> dict:
    """Daily usage totals over ``from_``..``to`` grouped by day, key, model or route (contract ``UsageReport``)."""
    if group_by is None:
        group_by = "day"
    if group_by not in GROUP_BY:
        raise ServiceError("invalid_request", f"group_by: must be one of {', '.join(GROUP_BY)}")
    start, end = _date_range(from_, to, today)
    email = _user_filter(viewer, user)
    where, args = ["u.day BETWEEN ? AND ?"], [start.isoformat(), end.isoformat()]
    if email is not None:
        where.append("k.owner_email = ?")
        args.append(email)
    with db.read() as conn:
        if key_id is not None:
            if not isinstance(key_id, str) or not KEY_ID_RE.fullmatch(key_id):
                raise ServiceError("invalid_request", "key_id: must be 12 lowercase base32 characters")
            _visible_key(conn, viewer, key_id)
            where.append("u.public_id = ?")
            args.append(key_id)
        dims = {
            "day": "u.day",
            "key": "u.public_id, k.name, k.owner_email",
            "model": "u.model",
            "route": "u.route",
        }[group_by]
        rows = conn.execute(
            f"SELECT {dims}, sum(u.requests) AS req, sum(u.client_errors) AS ce, sum(u.server_errors) AS se,"
            f" sum(u.latency_ms_sum) AS lat, sum(u.cost_usd_sum) AS cost FROM usage_daily u JOIN api_keys k ON k.public_id = u.public_id"
            f" WHERE {' AND '.join(where)} GROUP BY {dims} HAVING sum(u.requests) > 0",
            args,
        ).fetchall()
    out = []
    for r in rows:
        if group_by == "day":
            dim: dict[str, Any] = {"date": r["day"]}
        elif group_by == "key":
            dim = {"key_id": r["public_id"], "key_name": r["name"], "owner_email": r["owner_email"]}
        elif group_by == "model":
            dim = {"model": r["model"] or None}
        else:
            dim = {"route": r["route"]}
        out.append({**dim, **_totals(r["req"], r["ce"], r["se"], r["lat"], r["cost"])})
    if group_by == "day":
        out.sort(key=lambda row: row["date"])
    else:
        out.sort(key=lambda row: (-row["requests"], str(next(iter(row.values())) or "")))
    total = _totals(
        sum(r["req"] for r in rows),
        sum(r["ce"] for r in rows),
        sum(r["se"] for r in rows),
        sum(r["lat"] for r in rows),
        sum(r["cost"] for r in rows),
    )
    return {"from": start.isoformat(), "to": end.isoformat(), "group_by": group_by, "rows": out, "total": total}


def audit_query(
    db: Database,
    viewer: Viewer,
    from_: Any = None,
    to: Any = None,
    before: Any = None,
    limit: Any = 100,
    *,
    today: date | None = None,
) -> dict:
    """Owner-only audit entries in ``from_``..``to`` (UTC dates, inclusive), newest first, paged by ``before``."""
    _require_owner(viewer)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_AUDIT_LIMIT:
        raise ServiceError("invalid_request", f"limit: must be an integer from 1 to {MAX_AUDIT_LIMIT}")
    if before is not None and (isinstance(before, bool) or not isinstance(before, int) or before < 1):
        raise ServiceError("invalid_request", "before: must be a positive integer")
    start, end = _date_range(from_, to, today)
    where = ["at >= ?", "at < ?"]
    args: list[Any] = [f"{start.isoformat()}T00:00:00Z", f"{(end + timedelta(days=1)).isoformat()}T00:00:00Z"]
    if before is not None:
        where.append("id < ?")
        args.append(before)
    with db.read() as conn:
        rows = conn.execute(
            f"SELECT * FROM audit WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?", [*args, limit + 1]
        ).fetchall()
    more = len(rows) > limit
    entries = [
        {
            "id": r["id"],
            "at": r["at"],
            "event": r["event"],
            "actor_email": r["actor_email"],
            "target_email": r["target_email"],
            "key_id": r["key_id"],
            "detail": json.loads(r["detail"]),
        }
        for r in rows[:limit]
    ]
    return {"entries": entries, "next_before": entries[-1]["id"] if more else None}
