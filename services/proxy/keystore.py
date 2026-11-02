"""API key stores: who presented a valid ``Authorization: Bearer`` key.

- :class:`FileKeyStore` (default, ``KEYSTORE=file``): the keys from ``DECISION_API_KEYS_FILE`` /
  ``DECISION_API_KEYS``, compared in constant time against every configured key. Nothing is written.
- :class:`SqliteKeyStore` (``KEYSTORE=sqlite``): keys in SQLite, stored only as HMAC-SHA256(pepper, key).

Key format of issued keys: ``dk_<public_id>_<secret>``; ``public_id`` is 12 lowercase base32 characters
(the key id), ``secret`` is 32 random bytes, base64url without padding (43 characters).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from db import Database, parse_iso, utcnow

PUBLIC_ID_LEN = 12
SECRET_BYTES = 32
ISSUED_KEY_RE = re.compile(r"^dk_([a-z2-7]{12})_([A-Za-z0-9_-]{43})$")
SOURCE_FILE = "file"
SOURCE_ISSUED = "issued"
SOURCE_IMPORTED = "imported"


@dataclass(frozen=True)
class Principal:
    """The caller behind a valid key.

    ``key_id`` goes into the access log and the ``key_id`` metric label: the public id for issued keys, the
    8-hex sha256 prefix for file and imported keys (as before the SQLite store). ``public_id`` is the
    key's row in the database (None for file keys), used for usage accounting.
    """

    key_id: str
    public_id: str | None
    owner_email: str | None
    source: str


class KeyStore(Protocol):
    def authenticate(self, raw_key: str) -> Principal | None: ...


# ---------------------------------------------------------------- key format and hashing


def new_public_id() -> str:
    # 8 random bytes -> 13 base32 chars (no padding); keep 12 (60 bits)
    return base64.b32encode(secrets.token_bytes(8)).decode().lower()[:PUBLIC_ID_LEN]


def generate_key() -> tuple[str, str]:
    """A new issued key and its public id."""
    public_id = new_public_id()
    secret = base64.urlsafe_b64encode(secrets.token_bytes(SECRET_BYTES)).rstrip(b"=").decode()
    return f"dk_{public_id}_{secret}", public_id


def parse_key(raw: str) -> tuple[str, str] | None:
    """``(public_id, secret)`` of a well-formed issued key, else None."""
    m = ISSUED_KEY_RE.fullmatch(raw or "")
    return (m.group(1), m.group(2)) if m else None


def hash_key(pepper: bytes, raw: str) -> str:
    """What the database stores for a key: hex HMAC-SHA256 keyed with the server pepper."""
    return hmac.new(pepper, raw.encode(), hashlib.sha256).hexdigest()


def legacy_key_id(raw: str) -> str:
    """Non-reversible short id for logs and metrics of file and imported keys."""
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def bearer_token(header: str | None) -> str | None:
    """The token of an ``Authorization: Bearer <token>`` header, else None."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


# ---------------------------------------------------------------- file store


class FileKeyStore:
    """The configured keys (key file and env), unchanged from before the SQLite store."""

    def __init__(self, keys: Iterable[str]):
        self._keys = tuple(keys)

    def authenticate(self, raw_key: str) -> Principal | None:
        presented = raw_key.encode()
        found = None
        for key in self._keys:  # compare against every key; no early exit
            if hmac.compare_digest(presented, key.encode()):
                found = key
        if found is None:
            return None
        return Principal(key_id=legacy_key_id(found), public_id=None, owner_email=None, source=SOURCE_FILE)


# ---------------------------------------------------------------- SQLite store


@dataclass(frozen=True)
class _CachedKey:
    public_id: str
    key_hmac: str
    owner_email: str
    source: str
    expires_at: datetime | None


class SqliteKeyStore:
    """Keys from the ``api_keys`` table, cached in memory until the next write to the database.

    Issued keys are found by the public id in the key, then the HMAC is compared in constant time. Imported
    (legacy) keys have no public id in their value and are found by their HMAC. Revoked keys are not cached;
    expired keys (``expires_at <= now``) fail. All failures look the same to the caller.
    """

    def __init__(self, db: Database, pepper: bytes, now: Callable[[], datetime] = utcnow):
        self._db = db
        self._pepper = pepper
        self._now = now
        self._lock = threading.Lock()
        self._generation: int | None = None
        self._by_public_id: dict[str, _CachedKey] = {}
        self._by_hmac: dict[str, _CachedKey] = {}

    def _load(self) -> tuple[dict[str, _CachedKey], dict[str, _CachedKey]]:
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT public_id, key_hmac, owner_email, source, expires_at FROM api_keys WHERE revoked_at IS NULL"
            ).fetchall()
        by_id, by_hmac = {}, {}
        for r in rows:
            key = _CachedKey(
                r["public_id"],
                r["key_hmac"],
                r["owner_email"],
                r["source"],
                parse_iso(r["expires_at"]) if r["expires_at"] else None,
            )
            if key.source == SOURCE_ISSUED:
                by_id[key.public_id] = key
            else:
                by_hmac[key.key_hmac] = key
        return by_id, by_hmac

    def _cache(self) -> tuple[dict[str, _CachedKey], dict[str, _CachedKey]]:
        with self._lock:
            generation = self._db.generation
            if self._generation != generation:
                self._by_public_id, self._by_hmac = self._load()
                self._generation = generation
            return self._by_public_id, self._by_hmac

    def authenticate(self, raw_key: str) -> Principal | None:
        if not raw_key:
            return None
        by_id, by_hmac = self._cache()
        digest = hash_key(self._pepper, raw_key)
        parsed = parse_key(raw_key)
        key = by_id.get(parsed[0]) if parsed else None
        if key is not None and not hmac.compare_digest(digest, key.key_hmac):
            key = None
        if key is None:
            key = by_hmac.get(digest)  # keyed hash: the lookup reveals nothing about the stored keys
        if key is None or (key.expires_at is not None and key.expires_at <= self._now()):
            return None
        kid = key.public_id if key.source == SOURCE_ISSUED else legacy_key_id(raw_key)
        return Principal(key_id=kid, public_id=key.public_id, owner_email=key.owner_email, source=key.source)
