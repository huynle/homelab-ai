"""Key format, HMAC storage, FileKeyStore and SqliteKeyStore authentication, legacy import."""

from __future__ import annotations

import hashlib
import hmac
import re

import pytest

import keyservice as ks
import keystore
from conftest import ALICE, OWNER, PEPPER
from keystore import FileKeyStore, Principal, SqliteKeyStore

KEY_RE = re.compile(r"^dk_[a-z2-7]{12}_[A-Za-z0-9_-]{43}$")


# ---------------------------------------------------------------- key format


def test_generated_key_matches_contract_format():
    raw, public_id = keystore.generate_key()
    assert KEY_RE.fullmatch(raw), raw
    assert raw.startswith(f"dk_{public_id}_")
    assert len({keystore.generate_key()[0] for _ in range(50)}) == 50


def test_public_id_is_12_lowercase_base32():
    for _ in range(50):
        assert re.fullmatch(r"[a-z2-7]{12}", keystore.new_public_id())


def test_parse_key_roundtrip():
    raw, public_id = keystore.generate_key()
    parsed = keystore.parse_key(raw)
    assert parsed is not None and parsed[0] == public_id and raw.endswith(parsed[1])


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "dk_",
        "dk_abc_def",
        "dk_ABCDEFGHIJKL_" + "a" * 43,  # upper case public id
        "dk_abcdefghijk1_" + "a" * 43,  # '1' is not base32
        "dk_abcdefghijkl_" + "a" * 42,  # short secret
        "dk_abcdefghijkl_" + "a" * 44,  # long secret
        "dk_abcdefghijkl_" + "a" * 42 + "=",  # padding
        "dk_abcdefghijkl_" + "a" * 42 + "+",  # not url-safe
        "xk_abcdefghijkl_" + "a" * 43,
        "test-key-0123456789abcdef",
    ],
)
def test_parse_key_rejects_malformed(raw):
    assert keystore.parse_key(raw) is None


def test_hash_key_is_hmac_sha256_hex():
    assert keystore.hash_key(PEPPER, "abc") == hmac.new(PEPPER, b"abc", hashlib.sha256).hexdigest()


def test_legacy_key_id_is_8_hex_sha256_prefix():
    assert keystore.legacy_key_id("abc") == hashlib.sha256(b"abc").hexdigest()[:8]


@pytest.mark.parametrize(
    "header,token",
    [
        ("Bearer abc", "abc"),
        ("bearer   abc  ", "abc"),
        ("Basic abc", None),
        ("Bearer", None),
        ("Bearer ", None),
        ("", None),
        (None, None),
    ],
)
def test_bearer_token(header, token):
    assert keystore.bearer_token(header) == token


# ---------------------------------------------------------------- FileKeyStore


def test_file_store_matches_configured_keys():
    store = FileKeyStore(("k-one", "k-two"))
    p = store.authenticate("k-two")
    assert p == Principal(key_id=keystore.legacy_key_id("k-two"), public_id=None, owner_email=None, source="file")
    assert store.authenticate("k-three") is None
    assert store.authenticate("") is None


def test_file_store_compares_every_key_without_early_exit(monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(keystore.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    FileKeyStore(("a", "b", "c")).authenticate("a")
    assert len(calls) == 3


# ---------------------------------------------------------------- SqliteKeyStore


@pytest.fixture
def store(owner_db, clock):
    return SqliteKeyStore(owner_db, PEPPER, now=clock)


def _alice(db):
    return ks.get_viewer(db, ALICE)


def test_issued_key_authenticates_with_public_id(store, owner_db, clock):
    created = ks.create_key(owner_db, _alice(owner_db), "laptop", now=clock())
    p = store.authenticate(created["key"])
    assert p == Principal(key_id=created["id"], public_id=created["id"], owner_email=ALICE, source="issued")


def test_unknown_and_malformed_keys_fail(store, owner_db, clock):
    created = ks.create_key(owner_db, _alice(owner_db), "laptop", now=clock())
    raw = created["key"]
    forged = raw[:-1] + ("A" if raw[-1] != "A" else "B")  # right public id, wrong secret
    assert store.authenticate(forged) is None
    other_id = "dk_" + keystore.new_public_id() + raw[15:]
    assert store.authenticate(other_id) is None
    for bad in ["", "dk_", "nonsense", raw + "x", raw.upper()]:
        assert store.authenticate(bad) is None


def test_revoked_key_fails_immediately(store, owner_db, clock):
    alice = _alice(owner_db)
    created = ks.create_key(owner_db, alice, "laptop", now=clock())
    assert store.authenticate(created["key"]) is not None  # warms the cache
    ks.revoke_key(owner_db, alice, created["id"], now=clock())
    assert store.authenticate(created["key"]) is None


def test_expired_key_fails(store, owner_db, clock):
    created = ks.create_key(
        owner_db, _alice(owner_db), "short", expires_at=clock() + clock.timedelta(hours=1), now=clock()
    )
    assert store.authenticate(created["key"]) is not None
    clock.advance(minutes=59, seconds=59)
    assert store.authenticate(created["key"]) is not None
    clock.advance(seconds=1)  # expires_at <= now
    assert store.authenticate(created["key"]) is None


def test_cache_is_used_between_writes(store, owner_db, clock, monkeypatch):
    created = ks.create_key(owner_db, _alice(owner_db), "laptop", now=clock())
    store.authenticate(created["key"])
    loads = []
    real = store._load
    monkeypatch.setattr(store, "_load", lambda: loads.append(1) or real())
    for _ in range(5):
        store.authenticate(created["key"])
    assert loads == []
    ks.rename_key(owner_db, _alice(owner_db), created["id"], "renamed", now=clock())
    store.authenticate(created["key"])
    assert loads == [1]


def test_sqlite_store_uses_constant_time_compare(store, owner_db, clock, monkeypatch):
    created = ks.create_key(owner_db, _alice(owner_db), "laptop", now=clock())
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(keystore.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    assert store.authenticate(created["key"]) is not None
    assert calls, "dk keys must be checked with hmac.compare_digest"


def test_raw_key_never_stored(tmp_path, owner_db, clock):
    created = ks.create_key(owner_db, _alice(owner_db), "laptop", now=clock())
    legacy = "legacy-sops-key-value-0123456789"
    ks.import_legacy_keys(owner_db, PEPPER, [legacy], OWNER, now=clock())
    owner_db.checkpoint()
    blob = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.is_file())
    secret = created["key"].split("_", 2)[2]
    for needle in (created["key"], secret, legacy):
        assert needle.encode() not in blob
    stored = {r[0] for r in owner_db.conn.execute("SELECT key_hmac FROM api_keys")}
    assert stored == {keystore.hash_key(PEPPER, created["key"]), keystore.hash_key(PEPPER, legacy)}


# ---------------------------------------------------------------- legacy import


def test_legacy_import_creates_owner_key_and_authenticates(store, owner_db, clock):
    imported = ks.import_legacy_keys(owner_db, PEPPER, ["legacy-1"], OWNER, now=clock())
    assert len(imported) == 1
    [key] = ks.list_keys(owner_db, ks.get_viewer(owner_db, OWNER), now=clock())
    assert key["source"] == "imported"
    assert key["prefix"] == "imported…"
    assert key["name"] == "imported (SOPS)"
    assert key["owner_email"] == OWNER
    assert key["rate_limit_rpm"] is None
    assert re.fullmatch(r"[a-z2-7]{12}", key["id"])
    p = store.authenticate("legacy-1")
    assert p == Principal(
        key_id=keystore.legacy_key_id("legacy-1"), public_id=key["id"], owner_email=OWNER, source="imported"
    )


def test_legacy_import_is_idempotent_and_names_several(owner_db, clock):
    ks.import_legacy_keys(owner_db, PEPPER, ["legacy-1", "legacy-2"], OWNER, now=clock())
    assert ks.import_legacy_keys(owner_db, PEPPER, ["legacy-1", "legacy-2"], OWNER, now=clock()) == []
    names = sorted(k["name"] for k in ks.list_keys(owner_db, ks.get_viewer(owner_db, OWNER), now=clock()))
    assert names == ["imported (SOPS)", "imported (SOPS) 2"]
    ks.import_legacy_keys(owner_db, PEPPER, ["legacy-3"], OWNER, now=clock())
    names = sorted(k["name"] for k in ks.list_keys(owner_db, ks.get_viewer(owner_db, OWNER), now=clock()))
    assert names == ["imported (SOPS)", "imported (SOPS) 2", "imported (SOPS) 3"]


def test_revoked_imported_key_is_not_reimported(store, owner_db, clock):
    owner = ks.get_viewer(owner_db, OWNER)
    [pid] = ks.import_legacy_keys(owner_db, PEPPER, ["legacy-1"], OWNER, now=clock())
    ks.revoke_key(owner_db, owner, pid, now=clock())
    assert ks.import_legacy_keys(owner_db, PEPPER, ["legacy-1"], OWNER, now=clock()) == []
    assert store.authenticate("legacy-1") is None


def test_legacy_key_shaped_like_dk_still_authenticates(store, owner_db, clock):
    raw, _ = keystore.generate_key()
    ks.import_legacy_keys(owner_db, PEPPER, [raw], OWNER, now=clock())
    p = store.authenticate(raw)
    assert p is not None and p.source == "imported"
