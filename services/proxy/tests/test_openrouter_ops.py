"""OpenRouter entries in /v1/models and /v1/health, and the OpenRouter settings."""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings
from conftest import API_KEY, IMAGE_URL, TEXT_URL
from orkit import OPENROUTER_KEY, FakeOpenRouter, write_key, write_registry


@pytest.fixture
def fake_or():
    return FakeOpenRouter()


def make_client(tmp_path, backends, fake_or, **overrides):
    base = {
        "api_keys": (API_KEY,),
        "text_backend_url": TEXT_URL,
        "image_backend_url": IMAGE_URL,
        "registry_file": write_registry(tmp_path),
        "openrouter_key_file": write_key(tmp_path),
    }
    base.update(overrides)
    app = create_app(Settings(**base), transport=httpx.MockTransport(fake_or.wrap(backends.handler)))
    return TestClient(app, raise_server_exceptions=False)


AUTH = {"Authorization": f"Bearer {API_KEY}"}


def test_models_lists_openrouter_entries_and_aliases(tmp_path, backends, fake_or):
    with make_client(tmp_path, backends, fake_or) as c:
        body = c.get("/v1/models", headers=AUTH).json()
    jev = next(m for m in body["models"] if m["id"] == "jev")
    assert jev == {
        "id": "jev",
        "task": "decisions",
        "backend": "openrouter",
        "version": "jev-1.14",
        "capabilities": ["text", "image", "box"],
        "paid": True,
        "backend_model": "typesafe/jev-1.14",
    }
    laya = next(m for m in body["models"] if m["id"] == "laya-english")
    assert laya["capabilities"] == ["text"] and laya["paid"] is False
    assert "backend_model" not in laya
    assert body["aliases"] == [
        {"id": "auto", "text": "laya-english", "image": "jev"},
        {"id": "vision", "text": "siglip2-base-patch16-224", "image": "siglip2-base-patch16-224"},
    ]
    assert OPENROUTER_KEY not in str(body)


def test_health_never_probes_openrouter_and_stays_ok(tmp_path, backends, fake_or):
    with make_client(tmp_path, backends, fake_or) as c:
        resp = c.get("/v1/health")
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert set(body["models"]) == {"laya-english", "laya-multilingual", "siglip2-base-patch16-224"}
    assert body["upstreams"] == {"jev": {"status": "not_probed", "configured": True}}
    assert fake_or.requests == []


def test_health_reports_missing_openrouter_key_without_degrading(tmp_path, backends, fake_or, caplog):
    caplog.set_level(logging.WARNING, logger="decision_api")
    with make_client(tmp_path, backends, fake_or, openrouter_key_file=None) as c:
        body = c.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["upstreams"] == {"jev": {"status": "not_probed", "configured": False}}
    assert any("DECISION_API_OPENROUTER_KEY_FILE" in r.getMessage() for r in caplog.records)


def test_health_without_openrouter_entries_has_no_upstreams(client):
    assert "upstreams" not in client.get("/v1/health").json()


def test_models_without_registry_aliases(client, auth):
    body = client.get("/v1/models", headers=auth).json()
    assert body["aliases"] == []
    assert all(m["paid"] is False for m in body["models"])


def test_backend_up_gauge_skips_openrouter(tmp_path, backends, fake_or):
    from metrics import generate_latest

    with make_client(tmp_path, backends, fake_or) as c:
        c.get("/v1/health")
        text = generate_latest(c.app.state.metrics.registry).decode()
    assert 'decision_api_backend_up{model="laya-english"} 1.0' in text
    assert 'decision_api_backend_up{model="jev"}' not in text


# ---------------------------------------------------------------- settings


def test_settings_read_openrouter_env():
    s = Settings.from_env(
        {
            "DECISION_API_KEYS": "k",
            "DECISION_API_OPENROUTER_KEY_FILE": " /run/secrets/openrouter ",
            "DECISION_API_OPENROUTER_REFERER": "https://example.test",
            "DECISION_API_FILE_KEYS_ALLOW_PAID": "1",
        }
    )
    assert s.openrouter_key_file == "/run/secrets/openrouter"
    assert s.openrouter_referer == "https://example.test"
    assert s.file_keys_allow_paid is True
    d = Settings.from_env({"DECISION_API_KEYS": "k"})
    assert d.openrouter_key_file is None
    assert d.openrouter_referer == "https://decide.ai.huynle.com"
    assert d.file_keys_allow_paid is False


def test_unreadable_openrouter_key_file_refuses_to_start(tmp_path, backends, fake_or):
    with pytest.raises(RuntimeError, match="DECISION_API_OPENROUTER_KEY_FILE"):
        make_client(tmp_path, backends, fake_or, openrouter_key_file=str(tmp_path / "missing"))


def test_empty_openrouter_key_file_refuses_to_start(tmp_path, backends, fake_or):
    with pytest.raises(RuntimeError, match="DECISION_API_OPENROUTER_KEY_FILE"):
        empty = tmp_path / "empty-key"
        empty.write_text("\n")
        make_client(tmp_path, backends, fake_or, openrouter_key_file=str(empty))
