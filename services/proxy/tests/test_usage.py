"""In-memory usage aggregation and its flush to usage_daily / last_used_at."""

from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

import keyservice as ks
from conftest import ALICE
from usage import UsageAggregator


@pytest.fixture
def key_id(owner_db, clock):
    return ks.create_key(owner_db, ks.get_viewer(owner_db, ALICE), "k", now=clock())["id"]


@pytest.fixture
def agg(owner_db, clock):
    return UsageAggregator(owner_db, now=clock)


def rows(db):
    return [
        tuple(r)
        for r in db.conn.execute(
            "SELECT day, public_id, route, model, requests, client_errors, server_errors, latency_ms_sum"
            " FROM usage_daily ORDER BY day, route, model"
        )
    ]


def last_used(db, pid):
    return db.conn.execute("SELECT last_used_at FROM api_keys WHERE public_id = ?", (pid,)).fetchone()[0]


def test_aggregates_by_day_key_route_model(agg, owner_db, key_id, clock):
    agg.record(key_id, "/v1/decide/text", "laya-english", 200, 10.0)
    agg.record(key_id, "/v1/decide/text", "laya-english", 200, 20.0)
    agg.record(key_id, "/v1/decide/text", None, 400, 1.5)
    agg.record(key_id, "/v1/decide/text", "laya-english", 502, 30.0)
    clock.advance(hours=6)  # 2026-11-03 00:00 UTC
    agg.record(key_id, "/v1/models", None, 200, 2.0)
    assert agg.flush() is True
    assert rows(owner_db) == [
        ("2026-11-02", key_id, "/v1/decide/text", "", 1, 1, 0, 1.5),
        ("2026-11-02", key_id, "/v1/decide/text", "laya-english", 3, 0, 1, 60.0),
        ("2026-11-03", key_id, "/v1/models", "", 1, 0, 0, 2.0),
    ]
    assert agg.pending() == {}


def test_flush_adds_to_existing_rows(agg, owner_db, key_id):
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    agg.flush()
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    agg.flush()
    assert rows(owner_db) == [("2026-11-02", key_id, "/v1/models", "", 2, 0, 0, 2.0)]


def test_empty_flush_writes_nothing(agg, owner_db):
    g = owner_db.generation
    assert agg.flush() is True
    assert rows(owner_db) == []
    assert owner_db.generation == g


def test_flush_failure_keeps_counts_and_retries(agg, owner_db, key_id, monkeypatch, caplog):
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    real = owner_db.write

    def broken(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(owner_db, "write", broken)
    assert agg.flush() is False
    assert "usage flush failed" in caplog.text
    agg.record(key_id, "/v1/models", None, 200, 1.0)  # recorded while the DB was failing
    monkeypatch.setattr(owner_db, "write", real)
    assert agg.flush() is True
    assert rows(owner_db) == [("2026-11-02", key_id, "/v1/models", "", 2, 0, 0, 2.0)]


def test_last_used_written_at_most_once_a_minute(agg, owner_db, key_id, clock):
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    agg.flush()
    assert last_used(owner_db, key_id) == "2026-11-02T18:00:00Z"
    clock.advance(seconds=30)
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    agg.flush()
    assert last_used(owner_db, key_id) == "2026-11-02T18:00:00Z"  # < 60 s since the last write
    clock.advance(seconds=31)
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    agg.flush()
    assert last_used(owner_db, key_id) == "2026-11-02T18:01:01Z"


def test_last_used_does_not_invalidate_key_cache(agg, owner_db, key_id):
    g = owner_db.generation
    agg.record(key_id, "/v1/models", None, 200, 1.0)
    agg.flush()
    assert owner_db.generation == g


def test_record_is_thread_safe(agg, owner_db, key_id):
    def work():
        for _ in range(500):
            agg.record(key_id, "/v1/models", None, 200, 1.0)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for _ in range(5):
        agg.flush()
    for t in threads:
        t.join()
    agg.flush()
    assert rows(owner_db)[0][4] == 4000


def test_run_loop_flushes_periodically_and_on_cancel(agg, owner_db, key_id):
    async def main():
        task = asyncio.create_task(agg.run(0.01))
        agg.record(key_id, "/v1/models", None, 200, 1.0)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if rows(owner_db):
                break
        assert rows(owner_db)
        agg.record(key_id, "/v1/models", None, 200, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    agg.flush()  # the lifespan's shutdown flush
    assert rows(owner_db)[0][4] == 2
