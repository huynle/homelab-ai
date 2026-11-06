"""Paid-model access per key (migration 0002, key stores, admin PATCH) and upstream cost in usage."""

from __future__ import annotations

import json
import shutil
import sqlite3

import pytest

import db as dbmod
import keyservice as ks
from conftest import ALICE, OWNER, PEPPER
from keystore import FileKeyStore, SqliteKeyStore
from usage import UsageAggregator


def columns(conn, table: str) -> dict[str, sqlite3.Row]:
    return {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------- migration 0002


def test_fresh_db_has_allow_paid_and_cost_columns(db):
    assert db.schema_version() == 2
    keys = columns(db.conn, "api_keys")
    assert keys["allow_paid"]["notnull"] == 1 and keys["allow_paid"]["dflt_value"] == "0"
    usage = columns(db.conn, "usage_daily")
    assert usage["cost_usd_sum"]["notnull"] == 1 and usage["cost_usd_sum"]["dflt_value"] == "0"


def test_migrates_a_version_1_db_keeping_rows(tmp_path):
    only_v1 = tmp_path / "migrations"
    only_v1.mkdir()
    shutil.copy(dbmod.MIGRATIONS / "0001_initial.sql", only_v1)
    path = tmp_path / "a.db"
    conn = dbmod.connect(path)
    dbmod.migrate(conn, only_v1)
    conn.execute("INSERT INTO users (email, role) VALUES ('o@example.com', 'owner')")
    conn.execute(
        "INSERT INTO api_keys (public_id, key_hmac, name, owner_email, source, prefix, created_at)"
        " VALUES ('aaaaaaaaaaaa', 'h', 'n', 'o@example.com', 'issued', 'p', 't')"
    )
    conn.execute(  # the version 1 columns (upsert_usage writes cost_usd_sum, which version 1 lacks)
        "INSERT INTO usage_daily (day, public_id, route, model, requests) VALUES"
        " ('2026-11-02', 'aaaaaaaaaaaa', '/v1/models', '', 3)"
    )
    conn.close()

    database = dbmod.Database.open(path)
    try:
        assert database.schema_version() == 2
        assert database.conn.execute("SELECT allow_paid FROM api_keys").fetchone()[0] == 0
        assert tuple(database.conn.execute("SELECT requests, cost_usd_sum FROM usage_daily").fetchone()) == (3, 0.0)
        with pytest.raises(sqlite3.IntegrityError), database.write() as c:
            c.execute("UPDATE api_keys SET allow_paid = 2")
    finally:
        database.close()


# ---------------------------------------------------------------- key stores


def test_file_keys_are_denied_paid_unless_enabled():
    assert FileKeyStore(["k1"]).authenticate("k1").allow_paid is False
    assert FileKeyStore(["k1"], allow_paid=True).authenticate("k1").allow_paid is True


def test_sqlite_principal_reflects_allow_paid_immediately(owner_db, clock):
    owner = ks.get_viewer(owner_db, OWNER)
    created = ks.create_key(owner_db, ks.get_viewer(owner_db, ALICE), "k", now=clock())
    assert created["allow_paid"] is False
    store = SqliteKeyStore(owner_db, PEPPER, now=clock)
    assert store.authenticate(created["key"]).allow_paid is False
    ks.set_paid_access(owner_db, owner, created["id"], True, now=clock())
    assert store.authenticate(created["key"]).allow_paid is True
    ks.set_paid_access(owner_db, owner, created["id"], False, now=clock())
    assert store.authenticate(created["key"]).allow_paid is False


def test_set_paid_access_is_owner_only_and_audited(owner_db, clock):
    owner, alice = ks.get_viewer(owner_db, OWNER), ks.get_viewer(owner_db, ALICE)
    key = ks.create_key(owner_db, alice, "k", now=clock())
    with pytest.raises(ks.ServiceError) as info:
        ks.set_paid_access(owner_db, alice, key["id"], True, now=clock())
    assert info.value.code == "forbidden"
    out = ks.set_paid_access(owner_db, owner, key["id"], True, now=clock())
    assert out["allow_paid"] is True
    ks.set_paid_access(owner_db, owner, key["id"], True, now=clock())  # no change: no second audit row
    rows = owner_db.conn.execute(
        "SELECT actor_email, target_email, key_id, detail FROM audit WHERE event = 'key.paid_access_changed'"
    ).fetchall()
    assert [(r[0], r[1], r[2], json.loads(r[3])) for r in rows] == [(OWNER, ALICE, key["id"], {"allow_paid": True})]
    assert ks.list_keys(owner_db, alice, now=clock())[0]["allow_paid"] is True
    ks.revoke_key(owner_db, owner, key["id"], now=clock())
    with pytest.raises(ks.ServiceError) as info:
        ks.set_paid_access(owner_db, owner, key["id"], False, now=clock())
    assert info.value.code == "conflict"


# ---------------------------------------------------------------- admin API


def test_admin_patch_allow_paid(admin):
    from test_admin import audit, invite

    invite(admin, ALICE)
    alice, owner = admin.signed_in(ALICE), admin.signed_in(OWNER)
    key = admin.call(alice, "POST", "/admin/api/keys", {"name": "a"}).json()
    url = f"/admin/api/keys/{key['id']}"

    resp = admin.call(alice, "PATCH", url, {"allow_paid": True})
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "forbidden"
    resp = admin.call(owner, "PATCH", url, {"allow_paid": True}, csrf=None)
    assert resp.status_code == 403 and resp.json()["error"]["code"] == "csrf_failed"
    resp = admin.call(owner, "PATCH", url, {"allow_paid": "yes"})
    assert resp.status_code == 400

    resp = admin.call(owner, "PATCH", url, {"allow_paid": True})
    assert resp.status_code == 200 and resp.json()["allow_paid"] is True
    assert audit(admin.db, "key.paid_access_changed") == [
        ("key.paid_access_changed", OWNER, ALICE, {"allow_paid": True})
    ]
    assert alice.get("/admin/api/keys").json()["keys"][0]["allow_paid"] is True
    # members can still rename their own key, and see allow_paid read-only
    resp = admin.call(alice, "PATCH", url, {"name": "renamed"})
    assert resp.status_code == 200 and resp.json() == dict(resp.json(), name="renamed", allow_paid=True)


def test_admin_ui_shows_paid_access(admin):
    invite_alice = ks.invite_member
    invite_alice(admin.db, ks.get_viewer(admin.db, OWNER), ALICE, now=admin.clock())
    alice, owner = admin.signed_in(ALICE), admin.signed_in(OWNER)
    key = admin.call(alice, "POST", "/admin/api/keys", {"name": "a"}).json()
    owner_page = owner.get("/admin/keys").text
    assert f'data-paid-key="{key["id"]}"' in owner_page
    member_page = alice.get("/admin").text
    assert "data-paid-key" not in member_page
    assert "Paid models" in member_page
    js = owner.get("/admin/static/admin.js").text
    assert "allow_paid" in js


# ---------------------------------------------------------------- usage cost


@pytest.fixture
def key_id(owner_db, clock):
    return ks.create_key(owner_db, ks.get_viewer(owner_db, ALICE), "k", now=clock())["id"]


def test_usage_aggregates_cost(owner_db, key_id, clock):
    agg = UsageAggregator(owner_db, now=clock)
    agg.record(key_id, "/api/alpha/decisions", "jev", 200, 10.0, cost_usd=0.00002)
    agg.record(key_id, "/api/alpha/decisions", "jev", 200, 10.0, cost_usd=0.00003)
    agg.record(key_id, "/api/alpha/decisions", "laya-english", 200, 5.0)
    assert agg.flush() is True
    rows = owner_db.conn.execute("SELECT model, requests, cost_usd_sum FROM usage_daily ORDER BY model").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("jev", 2), ("laya-english", 1)]
    assert rows[0][2] == pytest.approx(0.00005) and rows[1][2] == 0
    report = ks.usage_query(owner_db, ks.get_viewer(owner_db, OWNER), group_by="model", today=clock().date())
    by_model = {r["model"]: r for r in report["rows"]}
    assert by_model["jev"]["cost_usd"] == pytest.approx(0.00005)
    assert by_model["laya-english"]["cost_usd"] == 0
    assert report["total"]["cost_usd"] == pytest.approx(0.00005)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_usage_ignores_invalid_cost(owner_db, key_id, clock, bad):
    agg = UsageAggregator(owner_db, now=clock)
    agg.record(key_id, "/api/alpha/decisions", "jev", 200, 1.0, cost_usd=bad)
    agg.flush()
    assert owner_db.conn.execute("SELECT cost_usd_sum FROM usage_daily").fetchone()[0] == 0
