"""SQLite storage for the key store: connection settings, forward-only migrations, small shared helpers.

Used only when ``KEYSTORE=sqlite``. One process owns the database (a single replica), so one connection guarded
by a lock is enough; WAL mode lets the nightly online backup read while the app writes.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc  # noqa: UP017 (Python 3.10 has no datetime.UTC)
MIGRATIONS = Path(__file__).resolve().parent / "migrations"
MIGRATION_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
BUSY_TIMEOUT_MS = 5000


def iso(dt: datetime) -> str:
    """UTC timestamp as ``YYYY-MM-DDTHH:MM:SSZ`` (whole seconds; sorts lexically)."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime; pass an aware (UTC) datetime")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database in autocommit mode (transactions are explicit) with WAL, foreign keys and a busy timeout."""
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if mode != "wal":
        conn.close()
        raise RuntimeError(f"SQLite database {path} cannot use WAL mode (got {mode!r})")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def list_migrations(directory: Path = MIGRATIONS) -> list[tuple[int, Path]]:
    found = []
    for path in sorted(directory.iterdir()):
        m = MIGRATION_RE.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    versions = [v for v, _ in found]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"migrations in {directory} are not numbered 0001.. without gaps: {versions}")
    return found


def latest_version(directory: Path = MIGRATIONS) -> int:
    migrations = list_migrations(directory)
    return migrations[-1][0] if migrations else 0


def current_version(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT coalesce(max(version), 0) FROM schema_version").fetchone()[0]


def migrate(conn: sqlite3.Connection, directory: Path = MIGRATIONS) -> list[int]:
    """Apply pending migrations in order, each in its own transaction. Returns the versions applied.

    Forward-only: there are no down migrations, and a database newer than the code is refused.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    migrations = list_migrations(directory)
    have = current_version(conn)
    latest = migrations[-1][0] if migrations else 0
    if have > latest:
        raise RuntimeError(f"database schema version {have} is newer than this code ({latest}); refusing to downgrade")
    applied = []
    for version, path in migrations:
        if version <= have:
            continue
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{path.read_text()}\n;\n"
            f"INSERT INTO schema_version (version, applied_at) VALUES ({version:d}, '{iso(utcnow())}');\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        applied.append(version)
    return applied


def upsert_usage(
    conn: sqlite3.Connection,
    day: str,
    public_id: str,
    route: str,
    model: str | None,
    requests: int,
    client_errors: int,
    server_errors: int,
    latency_ms_sum: float,
) -> None:
    """Add to one usage_daily row. ``model`` None is stored as '' so the row's primary key stays unique."""
    conn.execute(
        "INSERT INTO usage_daily"
        " (day, public_id, route, model, requests, client_errors, server_errors, latency_ms_sum)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (day, public_id, route, model) DO UPDATE SET"
        "  requests = requests + excluded.requests,"
        "  client_errors = client_errors + excluded.client_errors,"
        "  server_errors = server_errors + excluded.server_errors,"
        "  latency_ms_sum = latency_ms_sum + excluded.latency_ms_sum",
        (day, public_id, route, model or "", requests, client_errors, server_errors, latency_ms_sum),
    )


class Database:
    """One connection, one lock. ``generation`` counts committed writes so caches know when to reload.

    ``pepper`` is the server secret keyed into every stored key's HMAC (from ``DECISION_API_KEY_PEPPER_FILE``).
    """

    def __init__(self, conn: sqlite3.Connection, path: str | Path, pepper: bytes | None = None):
        self.conn = conn
        self.path = Path(path)
        self.pepper = pepper  # HMAC key for stored API keys; never written to the database
        self.lock = threading.RLock()
        self.generation = 0

    @classmethod
    def open(cls, path: str | Path, pepper: bytes | None = None) -> Database:
        conn = connect(path)
        try:
            migrate(conn)
        except Exception:
            conn.close()
            raise
        return cls(conn, path, pepper)

    def schema_version(self) -> int:
        with self.lock:
            return current_version(self.conn)

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            yield self.conn

    @contextmanager
    def write(self, bump: bool = True) -> Iterator[sqlite3.Connection]:
        """One write transaction. ``bump=False`` for writes that do not change keys (usage, last_used_at)."""
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")
            if bump:
                self.generation += 1

    def checkpoint(self) -> None:
        with self.lock:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self.lock:
            self.conn.close()
