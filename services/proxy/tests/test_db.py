"""SQLite database: pragmas and forward-only numbered migrations."""

from __future__ import annotations

import sqlite3

import pytest

import db as dbmod

TABLES = {"schema_version", "users", "api_keys", "usage_daily", "sessions", "audit"}


def _tables(conn) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows} - {"sqlite_sequence"}


def test_migrates_empty_db_to_latest(tmp_path):
    database = dbmod.Database.open(tmp_path / "a.db")
    try:
        assert _tables(database.conn) == TABLES
        assert database.schema_version() == dbmod.latest_version() >= 1
    finally:
        database.close()


def test_rerun_is_a_noop(tmp_path):
    path = tmp_path / "a.db"
    dbmod.Database.open(path).close()
    database = dbmod.Database.open(path)
    try:
        rows = database.conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        assert [r[0] for r in rows] == list(range(1, dbmod.latest_version() + 1))
        assert dbmod.migrate(database.conn) == []
    finally:
        database.close()


def test_pragmas_wal_foreign_keys_busy_timeout(db):
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000


def test_migrations_are_numbered_and_contiguous():
    versions = [v for v, _ in dbmod.list_migrations()]
    assert versions == list(range(1, len(versions) + 1))


def test_refuses_db_newer_than_code(tmp_path):
    path = tmp_path / "a.db"
    dbmod.Database.open(path).close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (999, 'x')")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        dbmod.Database.open(path)


def test_failed_migration_rolls_back(tmp_path):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_ok.sql").write_text("CREATE TABLE a (x INTEGER);")
    (migrations / "0002_bad.sql").write_text("CREATE TABLE b (x INTEGER); THIS IS NOT SQL;")
    conn = dbmod.connect(tmp_path / "a.db")
    with pytest.raises(sqlite3.Error):
        dbmod.migrate(conn, migrations)
    assert conn.execute("SELECT max(version) FROM schema_version").fetchone()[0] == 1
    assert "b" not in _tables(conn)
    conn.close()


def test_foreign_key_enforced(db):
    with pytest.raises(sqlite3.IntegrityError), db.write() as conn:
        conn.execute(
            "INSERT INTO api_keys (public_id, key_hmac, name, owner_email, source, prefix, created_at)"
            " VALUES ('aaaaaaaaaaaa', 'h', 'n', 'nobody@example.com', 'issued', 'p', 't')"
        )


def test_usage_null_model_rows_merge(db):
    """A request without a model must upsert into one row, not one row per request (NULL != NULL in SQL)."""
    with db.write() as conn:
        conn.execute("INSERT INTO users (email, role) VALUES ('o@example.com', 'owner')")
        conn.execute(
            "INSERT INTO api_keys (public_id, key_hmac, name, owner_email, source, prefix, created_at)"
            " VALUES ('aaaaaaaaaaaa', 'h', 'n', 'o@example.com', 'issued', 'p', 't')"
        )
    for _ in range(2):
        with db.write() as conn:
            dbmod.upsert_usage(conn, "2026-11-02", "aaaaaaaaaaaa", "/v1/models", None, 1, 0, 0, 5.0)
    rows = db.conn.execute("SELECT requests, latency_ms_sum FROM usage_daily").fetchall()
    assert [tuple(r) for r in rows] == [(2, 10.0)]


def test_write_bumps_generation_unless_told_not_to(db):
    g = db.generation
    with db.write():
        pass
    assert db.generation == g + 1
    with db.write(bump=False):
        pass
    assert db.generation == g + 1


def test_write_rolls_back_on_error(db):
    with pytest.raises(ValueError), db.write() as conn:
        conn.execute("INSERT INTO users (email, role) VALUES ('x@example.com', 'member')")
        raise ValueError("boom")
    assert db.conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0
