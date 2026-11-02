"""Key service layer: keys, members, usage and audit, with owner/member rules (used by the admin UI)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

import db as dbmod
import keyservice as ks
from conftest import ALICE, BOB, OWNER, PEPPER
from keyservice import ServiceError

UTC = timezone.utc  # noqa: UP017 (3.10 has no datetime.UTC)


def owner(db):
    return ks.get_viewer(db, OWNER)


def alice(db):
    return ks.get_viewer(db, ALICE)


def bob(db):
    return ks.get_viewer(db, BOB)


def audit_events(db) -> list[str]:
    return [r[0] for r in db.conn.execute("SELECT event FROM audit ORDER BY id")]


def code(excinfo) -> str:
    return excinfo.value.code


# ---------------------------------------------------------------- users


def test_ensure_owners_upserts_owner_role(db, clock):
    ks.ensure_owners(db, ["Owner@Example.com "], now=clock())
    ks.ensure_owners(db, ["owner@example.com"], now=clock())
    assert ks.get_viewer(db, "OWNER@example.com") == ks.Viewer(email=OWNER, role="owner")
    assert db.conn.execute("SELECT count(*) FROM users").fetchone()[0] == 1


def test_ensure_owners_promotes_existing_member(owner_db, clock):
    ks.ensure_owners(owner_db, [OWNER, ALICE], now=clock())
    assert alice(owner_db).role == "owner"


def test_get_viewer_unknown_and_removed(owner_db, clock):
    assert ks.get_viewer(owner_db, "nobody@example.com") is None
    ks.remove_member(owner_db, owner(owner_db), ALICE, now=clock())
    assert ks.get_viewer(owner_db, ALICE) is None


def test_invite_member_rules(owner_db, clock):
    with pytest.raises(ServiceError) as e:
        ks.invite_member(owner_db, alice(owner_db), "carol@example.com", now=clock())
    assert code(e) == "forbidden"
    with pytest.raises(ServiceError) as e:
        ks.invite_member(owner_db, owner(owner_db), "ALICE@example.com", now=clock())
    assert code(e) == "conflict"
    with pytest.raises(ServiceError) as e:
        ks.invite_member(owner_db, owner(owner_db), OWNER, now=clock())
    assert code(e) == "conflict"
    for bad in ["", "no-at-sign", "a@", "@b", "a b@example.com", "x" * 250 + "@e.co"]:
        with pytest.raises(ServiceError) as e:
            ks.invite_member(owner_db, owner(owner_db), bad, now=clock())
        assert code(e) == "invalid_request"
    m = ks.invite_member(owner_db, owner(owner_db), " Carol@Example.com", now=clock())
    assert m == {
        "email": "carol@example.com",
        "role": "member",
        "invited_at": "2026-11-02T18:00:00Z",
        "invited_by": OWNER,
        "last_sign_in_at": None,
        "active_keys": 0,
    }


def test_list_members_owner_first_then_by_invite(owner_db, clock):
    ks.create_key(owner_db, alice(owner_db), "k", now=clock())
    with pytest.raises(ServiceError) as e:
        ks.list_members(owner_db, alice(owner_db))
    assert code(e) == "forbidden"
    members = ks.list_members(owner_db, owner(owner_db), now=clock())
    assert [m["email"] for m in members] == [OWNER, ALICE, BOB]
    assert members[0]["invited_at"] is None and members[0]["role"] == "owner"
    assert members[1]["active_keys"] == 1


def test_remove_member_revokes_keys_and_sessions(owner_db, clock):
    a = alice(owner_db)
    k1 = ks.create_key(owner_db, a, "one", now=clock())
    k2 = ks.create_key(owner_db, a, "two", now=clock())
    ks.revoke_key(owner_db, a, k2["id"], now=clock())
    bob_key = ks.create_key(owner_db, bob(owner_db), "bob", now=clock())
    with owner_db.write() as conn:
        for sid, email in [("s-alice", ALICE), ("s-bob", BOB)]:
            conn.execute(
                "INSERT INTO sessions (id_hash, email, csrf_token_hash, created_at, expires_at)"
                " VALUES (?, ?, 'c', '2026-11-02T18:00:00Z', '2026-11-03T06:00:00Z')",
                (sid, email),
            )
    clock.advance(minutes=5)
    before = len(audit_events(owner_db))
    result = ks.remove_member(owner_db, owner(owner_db), "Alice@Example.com", now=clock())
    assert result == {"email": ALICE, "revoked_keys": [k1["id"]]}
    sessions = [r[0] for r in owner_db.conn.execute("SELECT email FROM sessions")]
    assert sessions == [BOB]
    assert audit_events(owner_db)[before:] == ["key.revoked", "member.removed"]
    # keys stay visible to the owner, revoked
    keys = {k["id"]: k for k in ks.list_keys(owner_db, owner(owner_db), user=ALICE, now=clock())}
    assert keys[k1["id"]]["status"] == "revoked" and keys[k1["id"]]["revoked_at"] == "2026-11-02T18:05:00Z"
    assert ks.get_key(owner_db, owner(owner_db), bob_key["id"], now=clock())["status"] == "active"
    # re-invite: allowed, old keys stay revoked
    ks.invite_member(owner_db, owner(owner_db), ALICE, now=clock())
    assert ks.get_key(owner_db, alice(owner_db), k1["id"], now=clock())["status"] == "revoked"


def test_remove_member_rules(owner_db, clock):
    with pytest.raises(ServiceError) as e:
        ks.remove_member(owner_db, alice(owner_db), BOB, now=clock())
    assert code(e) == "forbidden"
    with pytest.raises(ServiceError) as e:
        ks.remove_member(owner_db, owner(owner_db), OWNER, now=clock())
    assert code(e) == "forbidden"
    with pytest.raises(ServiceError) as e:
        ks.remove_member(owner_db, owner(owner_db), "nobody@example.com", now=clock())
    assert code(e) == "not_found"
    ks.remove_member(owner_db, owner(owner_db), BOB, now=clock())
    with pytest.raises(ServiceError) as e:
        ks.remove_member(owner_db, owner(owner_db), BOB, now=clock())
    assert code(e) == "not_found"


# ---------------------------------------------------------------- keys


def test_create_key_returns_full_key_once_and_audits(owner_db, clock):
    created = ks.create_key(owner_db, alice(owner_db), "  home-assistant  ", now=clock())
    assert created["key"].startswith(f"dk_{created['id']}_")
    assert created["name"] == "home-assistant"
    assert created["prefix"] == f"dk_{created['id']}_…"
    assert created["source"] == "issued"
    assert created["status"] == "active"
    assert created["created_at"] == "2026-11-02T18:00:00Z"
    assert created["expires_at"] is None and created["revoked_at"] is None and created["last_used_at"] is None
    assert created["rate_limit_rpm"] is None
    listed = ks.get_key(owner_db, alice(owner_db), created["id"], now=clock())
    assert "key" not in listed
    row = owner_db.conn.execute("SELECT event, actor_email, target_email, key_id, detail FROM audit").fetchall()[-1]
    assert tuple(row) == ("key.created", ALICE, ALICE, created["id"], '{"name": "home-assistant"}')


@pytest.mark.parametrize("name", ["", "   ", "x" * 65, None, 5])
def test_create_key_name_rules(owner_db, clock, name):
    with pytest.raises(ServiceError) as e:
        ks.create_key(owner_db, alice(owner_db), name, now=clock())
    assert code(e) == "invalid_request"


def test_create_key_name_64_ok(owner_db, clock):
    assert ks.create_key(owner_db, alice(owner_db), "x" * 64, now=clock())["name"] == "x" * 64


def test_create_key_expiry_rules(owner_db, clock):
    now = clock()
    a = alice(owner_db)
    for bad in [now, now - timedelta(seconds=1), now + timedelta(days=730, seconds=1), datetime(2027, 1, 1)]:
        with pytest.raises(ServiceError) as e:
            ks.create_key(owner_db, a, "k", expires_at=bad, now=now)
        assert code(e) == "invalid_request"
    ok = ks.create_key(owner_db, a, "k", expires_at=now + timedelta(days=730), now=now)
    assert ok["expires_at"] == "2028-11-01T18:00:00Z"
    later = ks.create_key(owner_db, a, "k", expires_at="2027-11-02T00:00:00Z", now=now)
    assert later["expires_at"] == "2027-11-02T00:00:00Z"
    with pytest.raises(ServiceError):
        ks.create_key(owner_db, a, "k", expires_at="not-a-date", now=now)


def test_member_limit_20_active_keys(owner_db, clock):
    a = alice(owner_db)
    keys = [ks.create_key(owner_db, a, f"k{i}", now=clock()) for i in range(20)]
    with pytest.raises(ServiceError) as e:
        ks.create_key(owner_db, a, "k21", now=clock())
    assert code(e) == "conflict"
    ks.revoke_key(owner_db, a, keys[0]["id"], now=clock())
    ks.create_key(owner_db, a, "k21", now=clock())  # revoked keys do not count


def test_expired_keys_do_not_count_toward_limit(owner_db, clock):
    a = alice(owner_db)
    for i in range(20):
        ks.create_key(owner_db, a, f"k{i}", expires_at=clock() + timedelta(hours=1), now=clock())
    clock.advance(hours=1)
    assert ks.create_key(owner_db, a, "fresh", now=clock())["status"] == "active"


def test_owner_has_no_key_limit(owner_db, clock):
    o = owner(owner_db)
    for i in range(22):
        ks.create_key(owner_db, o, f"k{i}", now=clock())


def test_list_keys_visibility(owner_db, clock):
    ka = ks.create_key(owner_db, alice(owner_db), "a", now=clock())
    clock.advance(seconds=1)
    kb = ks.create_key(owner_db, bob(owner_db), "b", now=clock())
    assert [k["id"] for k in ks.list_keys(owner_db, alice(owner_db), now=clock())] == [ka["id"]]
    assert [k["id"] for k in ks.list_keys(owner_db, alice(owner_db), user="ALICE@example.com", now=clock())] == [
        ka["id"]
    ]
    with pytest.raises(ServiceError) as e:
        ks.list_keys(owner_db, alice(owner_db), user=BOB, now=clock())
    assert code(e) == "forbidden"
    assert [k["id"] for k in ks.list_keys(owner_db, owner(owner_db), now=clock())] == [kb["id"], ka["id"]]
    assert [k["id"] for k in ks.list_keys(owner_db, owner(owner_db), user=BOB, now=clock())] == [kb["id"]]


def test_status_revoked_wins_over_expired(owner_db, clock):
    a = alice(owner_db)
    k = ks.create_key(owner_db, a, "k", expires_at=clock() + timedelta(hours=1), now=clock())
    clock.advance(hours=2)
    assert ks.get_key(owner_db, a, k["id"], now=clock())["status"] == "expired"
    ks.revoke_key(owner_db, a, k["id"], now=clock())
    assert ks.get_key(owner_db, a, k["id"], now=clock())["status"] == "revoked"


def test_others_keys_are_not_found_for_members(owner_db, clock):
    kb = ks.create_key(owner_db, bob(owner_db), "b", now=clock())
    a = alice(owner_db)
    for call in (
        lambda: ks.get_key(owner_db, a, kb["id"], now=clock()),
        lambda: ks.rename_key(owner_db, a, kb["id"], "mine", now=clock()),
        lambda: ks.revoke_key(owner_db, a, kb["id"], now=clock()),
        lambda: ks.get_key(owner_db, a, "zzzzzzzzzzzz", now=clock()),
    ):
        with pytest.raises(ServiceError) as e:
            call()
        assert code(e) == "not_found"
    assert ks.get_key(owner_db, bob(owner_db), kb["id"], now=clock())["status"] == "active"


def test_rename_key(owner_db, clock):
    a = alice(owner_db)
    k = ks.create_key(owner_db, a, "old", now=clock())
    assert ks.rename_key(owner_db, a, k["id"], " new ", now=clock())["name"] == "new"
    assert ks.rename_key(owner_db, a, k["id"], None, now=clock())["name"] == "new"  # empty body: no-op
    assert ks.rename_key(owner_db, owner(owner_db), k["id"], "by owner", now=clock())["name"] == "by owner"
    with pytest.raises(ServiceError) as e:
        ks.rename_key(owner_db, a, k["id"], "", now=clock())
    assert code(e) == "invalid_request"
    ks.revoke_key(owner_db, a, k["id"], now=clock())
    with pytest.raises(ServiceError) as e:
        ks.rename_key(owner_db, a, k["id"], "again", now=clock())
    assert code(e) == "conflict"


def test_revoke_is_idempotent_without_duplicate_audit(owner_db, clock):
    a = alice(owner_db)
    k = ks.create_key(owner_db, a, "k", now=clock())
    first = ks.revoke_key(owner_db, a, k["id"], now=clock())
    clock.advance(minutes=3)
    second = ks.revoke_key(owner_db, owner(owner_db), k["id"], now=clock())
    assert first["revoked_at"] == second["revoked_at"] == "2026-11-02T18:00:00Z"
    assert audit_events(owner_db).count("key.revoked") == 1
    detail = owner_db.conn.execute("SELECT detail FROM audit WHERE event = 'key.revoked'").fetchone()[0]
    assert detail == '{"reason": "user"}'


def test_owner_can_revoke_any_key(owner_db, clock):
    k = ks.create_key(owner_db, alice(owner_db), "k", now=clock())
    assert ks.revoke_key(owner_db, owner(owner_db), k["id"], now=clock())["status"] == "revoked"


# ---------------------------------------------------------------- usage


def _usage(db, rows):
    with db.write() as conn:
        for day, pid, route, model, req, ce, se, lat in rows:
            dbmod.upsert_usage(conn, day, pid, route, model, req, ce, se, lat)


@pytest.fixture
def usage_db(owner_db, clock):
    ka = ks.create_key(owner_db, alice(owner_db), "home-assistant", now=clock())["id"]
    kb = ks.create_key(owner_db, bob(owner_db), "bob-key", now=clock())["id"]
    [ko] = ks.import_legacy_keys(owner_db, PEPPER, ["legacy"], OWNER, now=clock())
    _usage(
        owner_db,
        [
            ("2026-10-03", ka, "/v1/decide/text", "laya-english", 1, 0, 0, 10.0),  # 30 days before today
            ("2026-10-04", ka, "/v1/decide/text", "laya-english", 2, 0, 0, 20.0),  # first day of the default
            ("2026-11-01", ka, "/v1/decide/text", "laya-english", 10, 1, 1, 1000.0),
            ("2026-11-01", ka, "/v1/decide/text", None, 3, 3, 0, 75.0),
            ("2026-11-02", ka, "/v1/classify/image", "siglip2-base-patch16-224", 4, 0, 0, 400.0),
            ("2026-11-02", kb, "/v1/models", None, 5, 0, 0, 50.0),
            ("2026-11-02", ko, "/v1/decide/text", "laya-english", 6, 0, 2, 600.0),
        ],
    )
    return owner_db, {"alice": ka, "bob": kb, "owner": ko}


TODAY = date(2026, 11, 2)


def test_usage_default_range_is_30_days_inclusive(usage_db):
    db, _ = usage_db
    r = ks.usage_query(db, owner(db), today=TODAY)
    assert (r["from"], r["to"], r["group_by"]) == ("2026-10-04", "2026-11-02", "day")
    assert [row["date"] for row in r["rows"]] == ["2026-10-04", "2026-11-01", "2026-11-02"]
    assert r["total"]["requests"] == 2 + 13 + 15


def test_usage_owner_by_day_totals(usage_db):
    db, _ = usage_db
    r = ks.usage_query(db, owner(db), from_="2026-11-01", to="2026-11-02", today=TODAY)
    assert r["rows"] == [
        {
            "date": "2026-11-01",
            "requests": 13,
            "errors": 5,
            "client_errors": 4,
            "server_errors": 1,
            "total_latency_ms": 1075.0,
            "avg_latency_ms": round(1075 / 13, 4),
        },
        {
            "date": "2026-11-02",
            "requests": 15,
            "errors": 2,
            "client_errors": 0,
            "server_errors": 2,
            "total_latency_ms": 1050.0,
            "avg_latency_ms": 70.0,
        },
    ]
    assert r["total"] == {
        "requests": 28,
        "errors": 7,
        "client_errors": 4,
        "server_errors": 3,
        "total_latency_ms": 2125.0,
        "avg_latency_ms": round(2125 / 28, 4),
    }


def test_usage_group_by_key_model_route(usage_db):
    db, ids = usage_db
    o = owner(db)
    by_key = ks.usage_query(db, o, from_="2026-11-01", to="2026-11-02", group_by="key", today=TODAY)["rows"]
    assert [(r["key_id"], r["key_name"], r["owner_email"], r["requests"]) for r in by_key] == [
        (ids["alice"], "home-assistant", ALICE, 17),
        (ids["owner"], "imported (SOPS)", OWNER, 6),
        (ids["bob"], "bob-key", BOB, 5),
    ]
    by_model = ks.usage_query(db, o, from_="2026-11-01", to="2026-11-02", group_by="model", today=TODAY)["rows"]
    assert [(r["model"], r["requests"]) for r in by_model] == [
        ("laya-english", 16),
        (None, 8),
        ("siglip2-base-patch16-224", 4),
    ]
    assert set(by_model[0]) == {"model", "requests", "errors", "client_errors", "server_errors"} | {
        "total_latency_ms",
        "avg_latency_ms",
    }
    by_route = ks.usage_query(db, o, from_="2026-11-01", to="2026-11-02", group_by="route", today=TODAY)["rows"]
    assert [(r["route"], r["requests"]) for r in by_route] == [
        ("/v1/decide/text", 19),
        ("/v1/models", 5),
        ("/v1/classify/image", 4),
    ]


def test_usage_member_sees_only_own(usage_db):
    db, ids = usage_db
    a = alice(db)
    r = ks.usage_query(db, a, from_="2026-11-01", to="2026-11-02", group_by="key", today=TODAY)
    assert [row["key_id"] for row in r["rows"]] == [ids["alice"]]
    assert ks.usage_query(db, a, user=ALICE, from_="2026-11-01", to="2026-11-02", today=TODAY)["total"]["requests"]
    with pytest.raises(ServiceError) as e:
        ks.usage_query(db, a, user=BOB, today=TODAY)
    assert code(e) == "forbidden"
    with pytest.raises(ServiceError) as e:
        ks.usage_query(db, a, key_id=ids["bob"], today=TODAY)
    assert code(e) == "not_found"


def test_usage_owner_filters(usage_db):
    db, ids = usage_db
    o = owner(db)
    r = ks.usage_query(db, o, from_="2026-11-01", to="2026-11-02", key_id=ids["bob"], today=TODAY)
    assert r["total"]["requests"] == 5
    r = ks.usage_query(db, o, from_="2026-11-01", to="2026-11-02", user=BOB, today=TODAY)
    assert r["total"]["requests"] == 5
    with pytest.raises(ServiceError) as e:
        ks.usage_query(db, o, key_id="zzzzzzzzzzzz", today=TODAY)
    assert code(e) == "not_found"


def test_usage_empty_total_has_null_avg(usage_db):
    db, _ = usage_db
    r = ks.usage_query(db, owner(db), from_="2025-01-01", to="2025-01-02", today=TODAY)
    assert r["rows"] == []
    assert r["total"]["requests"] == 0 and r["total"]["avg_latency_ms"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"from_": "2026-11-03", "to": "2026-11-02"},
        {"from_": "2025-11-01", "to": "2026-11-02"},  # 366 days apart: 367 days
        {"from_": "2026-13-01"},
        {"to": "yesterday"},
        {"group_by": "week"},
        {"key_id": "NOT-A-KEY"},
    ],
)
def test_usage_invalid_requests(usage_db, kwargs):
    db, _ = usage_db
    with pytest.raises(ServiceError) as e:
        ks.usage_query(db, owner(db), today=TODAY, **kwargs)
    assert code(e) == "invalid_request"


def test_usage_max_range_366_days_inclusive(usage_db):
    db, _ = usage_db
    r = ks.usage_query(db, owner(db), from_="2025-11-02", to="2026-11-02", today=TODAY)
    assert r["from"] == "2025-11-02"


# ---------------------------------------------------------------- audit


def test_audit_query_owner_only_newest_first_paged(owner_db, clock):
    a = alice(owner_db)
    for i in range(5):
        ks.create_key(owner_db, a, f"k{i}", now=clock())
    with pytest.raises(ServiceError) as e:
        ks.audit_query(owner_db, a, today=TODAY)
    assert code(e) == "forbidden"
    page = ks.audit_query(owner_db, owner(owner_db), limit=3, today=TODAY)
    ids = [e["id"] for e in page["entries"]]
    assert ids == sorted(ids, reverse=True) and len(ids) == 3
    assert page["next_before"] == ids[-1]
    entry = page["entries"][0]
    assert entry == {
        "id": ids[0],
        "at": "2026-11-02T18:00:00Z",
        "event": "key.created",
        "actor_email": ALICE,
        "target_email": ALICE,
        "key_id": entry["key_id"],
        "detail": {"name": "k4"},
    }
    rest = ks.audit_query(owner_db, owner(owner_db), before=page["next_before"], limit=100, today=TODAY)
    assert rest["next_before"] is None
    assert [e["event"] for e in rest["entries"]][-2:] == ["member.invited", "member.invited"]
    assert ks.audit_query(owner_db, owner(owner_db), from_="2026-11-03", to="2026-11-03", today=TODAY)["entries"] == []
    with pytest.raises(ServiceError):
        ks.audit_query(owner_db, owner(owner_db), limit=501, today=TODAY)


def test_audit_record_has_no_ip_or_token_columns(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(audit)")}
    assert cols == {"id", "at", "actor_email", "event", "target_email", "key_id", "detail"}


def test_record_sign_in_events(owner_db, clock):
    ks.record_audit(owner_db, "sign_in.denied", actor_email="mallory@example.com", detail={"reason": "not_invited"})
    with pytest.raises(ValueError):
        ks.record_audit(owner_db, "something.else")
    assert audit_events(owner_db)[-1] == "sign_in.denied"


def test_naive_now_is_rejected(owner_db):
    with pytest.raises(ValueError):
        ks.create_key(owner_db, alice(owner_db), "k", now=datetime(2026, 11, 2))


def test_iso_helpers():
    assert dbmod.iso(datetime(2026, 11, 2, 18, 0, 0, 123, tzinfo=UTC)) == "2026-11-02T18:00:00Z"
