"""Settings and API key loading."""

from __future__ import annotations

import os

import pytest

from app import create_app
from config import Settings, load_api_keys


def test_defaults():
    s = Settings.from_env({"DECISION_API_KEYS": "k1"})
    assert s.host == "127.0.0.1"
    assert s.port == 8080
    assert s.text_backend_url == "http://127.0.0.1:8000"
    assert s.text_timeout_s == 10.0
    assert s.image_timeout_s == 10.0
    assert s.health_timeout_s == 2.0
    assert s.laya_model_version == "laya-serve-0.4.20"
    assert s.image_model_id == "siglip2-base-patch16-224"
    assert s.image_model_version == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert s.api_keys == ("k1",)


def test_env_overrides():
    s = Settings.from_env(
        {
            "DECISION_API_KEYS": "k1",
            "HOST": "0.0.0.0",
            "PORT": "9090",
            "TEXT_BACKEND_URL": "http://laya:8000/",
            "IMAGE_BACKEND_URL": "http://img:9000",
            "TEXT_TIMEOUT_S": "3.5",
            "IMAGE_TIMEOUT_S": "30",
        }
    )
    assert (s.host, s.port) == ("0.0.0.0", 9090)
    assert s.text_backend_url == "http://laya:8000"  # trailing slash stripped
    assert s.image_backend_url == "http://img:9000"
    assert (s.text_timeout_s, s.image_timeout_s) == (3.5, 30.0)


def test_keys_from_file_and_env(tmp_path):
    f = tmp_path / "keys"
    f.write_text("# primary\nfile-key-1\n\n  file-key-2  # laptop\n#disabled-key\n")
    os.chmod(f, 0o600)
    keys = load_api_keys({"DECISION_API_KEYS_FILE": str(f), "DECISION_API_KEYS": "env-a, env-b,,file-key-1"})
    assert keys == ("file-key-1", "file-key-2", "env-a", "env-b")


def test_missing_key_file_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="DECISION_API_KEYS_FILE"):
        load_api_keys({"DECISION_API_KEYS_FILE": str(tmp_path / "nope")})


def test_refuses_to_start_without_keys():
    with pytest.raises(RuntimeError, match="no API keys"):
        create_app(Settings(api_keys=()))


def test_allow_no_auth_escape_hatch():
    s = Settings.from_env({"DECISION_API_ALLOW_NO_AUTH": "1"})
    assert s.api_keys == () and s.allow_no_auth is True
    create_app(s)  # does not raise


def test_settings_repr_hides_keys():
    s = Settings(api_keys=("super-secret-key",))
    assert "super-secret-key" not in repr(s)


def test_no_auth_mode_serves_without_header(backends):
    import httpx
    from fastapi.testclient import TestClient

    app = create_app(Settings(api_keys=(), allow_no_auth=True), transport=httpx.MockTransport(backends.handler))
    with TestClient(app) as c:
        assert c.get("/v1/models").status_code == 200
