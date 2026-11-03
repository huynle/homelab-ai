"""Key store wiring in the app: default file mode is unchanged and writes nothing; sqlite mode is opt-in."""

from __future__ import annotations

import json
import logging
import os
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

import keyservice as ks
import keystore
from app import create_app
from config import Settings
from conftest import ALICE, API_KEY, IMAGE_URL, OWNER, PEPPER, TEXT_URL

TEXT = "/v1/decide/text"


def _snapshot(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


# ---------------------------------------------------------------- default: file key store, nothing written


def test_default_mode_is_file_store_and_writes_nothing(tmp_path, monkeypatch, backends, laya_request, caplog):
    keys = tmp_path / "keys"
    keys.write_text(API_KEY + "\n")
    os.chmod(keys, 0o600)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    def no_sqlite(*a, **kw):
        raise AssertionError("sqlite3.connect called in the default key store mode")

    monkeypatch.setattr(sqlite3, "connect", no_sqlite)
    env = {
        "DECISION_API_KEYS_FILE": str(keys),
        "TEXT_BACKEND_URL": TEXT_URL,
        "IMAGE_BACKEND_URL": IMAGE_URL,
        # pepper/owner/flush alone must not enable anything; only KEYSTORE=sqlite does
        "DECISION_API_DB_PATH": str(work / "would-be.db"),
        "DECISION_API_OWNER_EMAILS": OWNER,
    }
    settings = Settings.from_env(env)
    assert settings.keystore == "file"
    before = _snapshot(tmp_path)
    caplog.set_level(logging.INFO, logger="decision_api.access")
    app = create_app(settings, transport=httpx.MockTransport(backends.handler))
    assert isinstance(app.state.keystore, keystore.FileKeyStore)
    assert app.state.db is None and app.state.usage is None
    with TestClient(app) as c:
        assert c.post(TEXT, json=laya_request, headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200
        assert c.post(TEXT, json=laya_request, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert _snapshot(tmp_path) == before
    entries = [json.loads(r.getMessage()) for r in caplog.records if r.name == "decision_api.access"]
    assert [e["key_id"] for e in entries] == [keystore.legacy_key_id(API_KEY), None]


def test_keystore_setting_values():
    assert Settings.from_env({"DECISION_API_KEYS": "k"}).keystore == "file"
    assert Settings.from_env({"DECISION_API_KEYS": "k", "KEYSTORE": "FILE"}).keystore == "file"
    with pytest.raises(ValueError, match="KEYSTORE"):
        Settings.from_env({"DECISION_API_KEYS": "k", "KEYSTORE": "postgres"})


def test_sqlite_settings_from_env(tmp_path):
    s = Settings.from_env(
        {
            "KEYSTORE": "sqlite",
            "DECISION_API_DB_PATH": str(tmp_path / "d.db"),
            "DECISION_API_KEY_PEPPER_FILE": str(tmp_path / "pepper"),
            "DECISION_API_OWNER_EMAILS": " Owner@Example.com, second@example.com ,",
            "DECISION_API_USAGE_FLUSH_S": "5",
        }
    )
    assert s.keystore == "sqlite"
    assert s.db_path == str(tmp_path / "d.db")
    assert s.pepper_file == str(tmp_path / "pepper")
    assert s.owner_emails == ("owner@example.com", "second@example.com")
    assert s.usage_flush_s == 5.0
    assert Settings.from_env({"DECISION_API_KEYS": "k"}).usage_flush_s == 15.0


# ---------------------------------------------------------------- sqlite mode


@pytest.fixture
def pepper_file(tmp_path):
    p = tmp_path / "pepper"
    p.write_bytes(PEPPER + b"\n")
    os.chmod(p, 0o600)
    return p


@pytest.fixture
def sqlite_settings(tmp_path, pepper_file):
    return Settings(
        api_keys=(API_KEY,),  # the SOPS key: imported on startup
        text_backend_url=TEXT_URL,
        image_backend_url=IMAGE_URL,
        keystore="sqlite",
        db_path=str(tmp_path / "decision-api.db"),
        pepper_file=str(pepper_file),
        owner_emails=(OWNER,),
        usage_flush_s=3600,
    )


@pytest.fixture
def sqlite_app(sqlite_settings, backends):
    app = create_app(sqlite_settings, transport=httpx.MockTransport(backends.handler))
    yield app
    app.state.db.close()


def _usage_rows(db):
    return [
        tuple(r)
        for r in db.conn.execute(
            "SELECT public_id, route, model, requests, client_errors, server_errors FROM usage_daily ORDER BY route"
        )
    ]


def test_sqlite_mode_issued_key_end_to_end(sqlite_app, laya_request, caplog):
    db = sqlite_app.state.db
    ks.invite_member(db, ks.get_viewer(db, OWNER), ALICE)
    created = ks.create_key(db, ks.get_viewer(db, ALICE), "laptop")
    auth = {"Authorization": f"Bearer {created['key']}"}
    caplog.set_level(logging.INFO, logger="decision_api.access")
    with TestClient(sqlite_app) as c:
        assert c.post(TEXT, json=laya_request, headers=auth).status_code == 200
        assert c.post(TEXT, json={"state": "x"}, headers=auth).status_code == 400
        assert c.get("/v1/models", headers={"Authorization": "Bearer dk_bad"}).status_code == 401
        mx = c.app.state.metrics.registry
        labels = {"route": TEXT, "model": "laya-english", "code": "200", "key_id": created["id"]}
        assert mx.get_sample_value("decision_api_requests_total", labels) == 1
        assert _usage_rows(db) == []  # flushed only on the interval or at shutdown
    entries = [json.loads(r.getMessage()) for r in caplog.records if r.name == "decision_api.access"]
    assert [e["key_id"] for e in entries] == [created["id"], created["id"], None]
    assert created["key"] not in caplog.text
    # shutdown flush; the 401 is not counted
    assert _usage_rows(db) == [(created["id"], TEXT, "", 1, 1, 0), (created["id"], TEXT, "laya-english", 1, 0, 0)]
    assert ks.get_key(db, ks.get_viewer(db, ALICE), created["id"])["last_used_at"] is not None


def test_sqlite_mode_revocation_is_immediate(sqlite_app):
    db = sqlite_app.state.db
    owner = ks.get_viewer(db, OWNER)
    created = ks.create_key(db, owner, "k")
    auth = {"Authorization": f"Bearer {created['key']}"}
    with TestClient(sqlite_app) as c:
        assert c.get("/v1/models", headers=auth).status_code == 200
        ks.revoke_key(db, owner, created["id"])
        resp = c.get("/v1/models", headers=auth)
        assert resp.status_code == 401
        assert resp.json()["error"] == {
            "code": "unauthorized",
            "message": "missing or invalid API key",
            "type": "authentication_error",
            "param": None,
        }


def test_sqlite_mode_imports_sops_key_keeping_hash_key_id(sqlite_app, sqlite_settings, backends, caplog):
    db = sqlite_app.state.db
    [key] = ks.list_keys(db, ks.get_viewer(db, OWNER))
    assert (key["source"], key["name"], key["owner_email"]) == ("imported", "imported (SOPS)", OWNER)
    caplog.set_level(logging.INFO, logger="decision_api.access")
    with TestClient(sqlite_app) as c:
        assert c.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200
    entry = json.loads([r for r in caplog.records if r.name == "decision_api.access"][0].getMessage())
    assert entry["key_id"] == keystore.legacy_key_id(API_KEY)
    assert _usage_rows(db) == [(key["id"], "/v1/models", "", 1, 0, 0)]
    # restart: no duplicate import
    again = create_app(sqlite_settings, transport=httpx.MockTransport(backends.handler))
    assert len(ks.list_keys(again.state.db, ks.get_viewer(again.state.db, OWNER))) == 1
    again.state.db.close()


def test_sqlite_mode_starts_with_no_file_keys(sqlite_settings, backends):
    s = Settings(**{**sqlite_settings.__dict__, "api_keys": ()})
    app = create_app(s, transport=httpx.MockTransport(backends.handler))
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 401  # auth always required in sqlite mode
    app.state.db.close()


def test_periodic_usage_flush(sqlite_settings, backends):
    import time

    s = Settings(**{**sqlite_settings.__dict__, "usage_flush_s": 0.02})
    app = create_app(s, transport=httpx.MockTransport(backends.handler))
    with TestClient(app) as c:
        assert c.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200
        deadline = time.monotonic() + 3
        while not _usage_rows(app.state.db) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _usage_rows(app.state.db)
    app.state.db.close()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"pepper_file": None}, "DECISION_API_KEY_PEPPER_FILE"),
        ({"pepper_file": "/nonexistent/pepper"}, "DECISION_API_KEY_PEPPER_FILE"),
        ({"db_path": None}, "DECISION_API_DB_PATH"),
        ({"db_path": "/nonexistent-dir/x.db"}, "DECISION_API_DB_PATH"),
        ({"owner_emails": ()}, "DECISION_API_OWNER_EMAILS"),
    ],
)
def test_sqlite_mode_refuses_to_start(sqlite_settings, change, match):
    with pytest.raises(RuntimeError, match=match):
        create_app(Settings(**{**sqlite_settings.__dict__, **change}))


def test_sqlite_mode_refuses_short_pepper(sqlite_settings, tmp_path):
    short = tmp_path / "short"
    short.write_bytes(b"x" * 31 + b"\n")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        create_app(Settings(**{**sqlite_settings.__dict__, "pepper_file": str(short)}))
    assert not (tmp_path / "decision-api.db").exists()


def test_sqlite_mode_refusal_messages_hide_pepper(sqlite_settings, tmp_path):
    short = tmp_path / "short"
    short.write_bytes(b"secret-pepper-bytes")
    with pytest.raises(RuntimeError) as e:
        create_app(Settings(**{**sqlite_settings.__dict__, "pepper_file": str(short)}))
    assert "secret-pepper-bytes" not in str(e.value)


def test_settings_repr_has_no_pepper(sqlite_settings):
    assert PEPPER.decode() not in repr(sqlite_settings)
