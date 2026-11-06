"""Registry-driven routing: model selection, defaults, aliases, per-model backends, health and /healthz."""

from __future__ import annotations

import copy
import json
import logging
import textwrap

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings
from conftest import API_KEY
from registry import RegistryError

TEXT = "/v1/decide/text"
IMAGE = "/v1/classify/image"

LAYA_A = "http://laya-a.test:8000"
LAYA_B = "http://laya-b.test:8100"
IMG_A = "http://img-a.test:8010"
IMG_B = "http://img-b.test:8020"

REGISTRY = f"""
defaults: {{text: laya-english, image: siglip2-base-patch16-224}}
models:
  - {{id: laya-english, task: text-decision, adapter: laya, backend_url: "{LAYA_A}", backend_model: english,
     version: laya-serve-0.4.20, timeout_s: 7}}
  - {{id: laya-multilingual, task: text-decision, adapter: laya, backend_url: "{LAYA_A}",
     backend_model: multilingual, version: laya-serve-0.4.20}}
  - {{id: laya-next, task: text-decision, adapter: laya, backend_url: "{LAYA_B}", version: laya-serve-0.5.0}}
  - {{id: siglip2-base-patch16-224, task: image-zero-shot, adapter: zero-shot-image, backend: siglip2,
     backend_url: "{IMG_A}", version: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2}}
  - {{id: siglip2-large, task: image-zero-shot, adapter: zero-shot-image, backend_url: "{IMG_B}",
     version: large-commit}}
"""


class Fleet:
    """Fake backends keyed by base URL. Each records requests; health and predict are programmable."""

    def __init__(self, laya_predict: dict, siglip_response: dict):
        self.requests: list[httpx.Request] = []
        self.down: set[str] = set()
        self.laya_predict = laya_predict
        self.siglip_response = siglip_response
        self.raise_for: dict[str, Exception] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if base in self.raise_for:
            raise self.raise_for[base]
        if request.url.path == "/health":
            if base in self.down:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json={"status": "ok"})
        if base in (LAYA_A, LAYA_B) and request.url.path == "/predict":
            return httpx.Response(200, json=copy.deepcopy(self.laya_predict))
        if base in (IMG_A, IMG_B) and request.url.path == "/classify":
            return httpx.Response(200, json=copy.deepcopy(self.siglip_response))
        raise AssertionError(f"unexpected backend url {request.url}")

    def posts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]


@pytest.fixture
def fleet(laya_predict, siglip_response) -> Fleet:
    return Fleet(laya_predict, siglip_response)


@pytest.fixture
def registry_file(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(REGISTRY))
    return path


@pytest.fixture
def rclient(registry_file, fleet):
    settings = Settings(api_keys=(API_KEY,), registry_file=str(registry_file))
    app = create_app(settings, transport=httpx.MockTransport(fleet.handler))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def sent_json(fleet: Fleet) -> dict:
    (req,) = fleet.posts()
    return json.loads(req.content)


# ---------------------------------------------------------------- text


def test_text_default_uses_default_backend_and_keeps_laya_routing(rclient, auth, laya_request, fleet):
    resp = rclient.post(TEXT, json=laya_request, headers=auth)
    assert resp.status_code == 200, resp.text
    (req,) = fleet.posts()
    assert str(req.url) == f"{LAYA_A}/predict"
    assert "model" not in sent_json(fleet)  # unpinned: Laya routes by script/language
    body = resp.json()
    assert body["model"] == "laya-english"  # routing.model in the capture is "english"
    assert body["model_version"] == "laya-serve-0.4.20"


def test_text_model_id_selects_entry_and_forwards_backend_model(rclient, auth, laya_request, fleet):
    resp = rclient.post(TEXT, json=dict(laya_request, model="laya-multilingual"), headers=auth)
    assert resp.status_code == 200, resp.text
    assert sent_json(fleet)["model"] == "multilingual"
    assert resp.json()["model"] == "laya-multilingual"


def test_text_entry_on_other_backend_without_backend_model(rclient, auth, laya_request, fleet):
    resp = rclient.post(TEXT, json=dict(laya_request, model="laya-next"), headers=auth)
    assert resp.status_code == 200, resp.text
    (req,) = fleet.posts()
    assert str(req.url) == f"{LAYA_B}/predict"
    assert "model" not in json.loads(req.content)
    body = resp.json()
    # routing says "english" but laya-english lives on another backend, so the selected entry is reported
    assert (body["model"], body["model_version"]) == ("laya-next", "laya-serve-0.5.0")


@pytest.mark.parametrize(
    "alias,forwarded,model", [("english", "english", "laya-english"), ("multi", "multilingual", "laya-multilingual")]
)
def test_text_laya_alias_still_works(rclient, auth, laya_request, fleet, alias, forwarded, model):
    resp = rclient.post(TEXT, json=dict(laya_request, model=alias), headers=auth)
    assert resp.status_code == 200, resp.text
    assert sent_json(fleet)["model"] == forwarded
    assert resp.json()["model"] == model


def test_text_per_model_timeout(rclient, auth, laya_request, fleet):
    fleet.raise_for[LAYA_A] = httpx.ReadTimeout("slow")
    resp = rclient.post(TEXT, json=laya_request, headers=auth)
    assert resp.status_code == 504
    assert "7000 ms" in resp.json()["error"]["message"]


def test_text_wrong_task_model_is_400(rclient, auth, laya_request, fleet):
    resp = rclient.post(TEXT, json=dict(laya_request, model="siglip2-large"), headers=auth)
    assert resp.status_code == 400
    message = resp.json()["error"]["message"]
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "image-zero-shot" in message
    assert "laya-english, laya-multilingual, laya-next" in message
    assert fleet.requests == []


def test_text_unknown_model_is_400(rclient, auth, laya_request, fleet):
    resp = rclient.post(TEXT, json=dict(laya_request, model="typed-decisions"), headers=auth)
    assert resp.status_code == 400
    assert "unknown model 'typed-decisions'" in resp.json()["error"]["message"]
    assert fleet.requests == []


# ---------------------------------------------------------------- image


def test_image_default_model(rclient, auth, image_request, fleet):
    resp = rclient.post(IMAGE, json=image_request, headers=auth)
    assert resp.status_code == 200, resp.text
    (req,) = fleet.posts()
    assert str(req.url) == f"{IMG_A}/classify"
    assert resp.json()["model"] == "siglip2-base-patch16-224"


def test_image_model_selects_backend_and_is_not_forwarded(rclient, auth, image_request, fleet):
    fleet.siglip_response.pop("model_version")
    resp = rclient.post(IMAGE, json=dict(image_request, model="siglip2-large"), headers=auth)
    assert resp.status_code == 200, resp.text
    (req,) = fleet.posts()
    assert str(req.url) == f"{IMG_B}/classify"
    assert json.loads(req.content) == image_request  # no `model` key
    body = resp.json()
    assert (body["model"], body["model_version"]) == ("siglip2-large", "large-commit")


def test_image_error_carries_selected_model(rclient, auth, image_request, fleet):
    fleet.raise_for[IMG_B] = httpx.ConnectError("refused")
    body = rclient.post(IMAGE, json=dict(image_request, model="siglip2-large"), headers=auth).json()
    assert body["error"]["code"] == "backend_unavailable"
    assert (body["model"], body["model_version"]) == ("siglip2-large", "large-commit")


@pytest.mark.parametrize("model", ["laya-english", "nope"])
def test_image_wrong_or_unknown_model_is_400(rclient, auth, image_request, fleet, model):
    resp = rclient.post(IMAGE, json=dict(image_request, model=model), headers=auth)
    assert resp.status_code == 400
    assert "siglip2-base-patch16-224, siglip2-large" in resp.json()["error"]["message"]
    assert fleet.requests == []


# ---------------------------------------------------------------- listing and health


def test_models_lists_registry_entries(rclient, auth):
    TEXT = {"capabilities": ["text"], "paid": False}
    IMAGE = {"capabilities": ["image"], "paid": False}
    assert rclient.get("/v1/models", headers=auth).json() == {
        "models": [
            {"id": "laya-english", "task": "text-decision", "backend": "laya", "version": "laya-serve-0.4.20", **TEXT},
            {
                "id": "laya-multilingual",
                "task": "text-decision",
                "backend": "laya",
                "version": "laya-serve-0.4.20",
                **TEXT,
            },
            {"id": "laya-next", "task": "text-decision", "backend": "laya", "version": "laya-serve-0.5.0", **TEXT},
            {
                "id": "siglip2-base-patch16-224",
                "task": "image-zero-shot",
                "backend": "siglip2",
                "version": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                **IMAGE,
            },
            {
                "id": "siglip2-large",
                "task": "image-zero-shot",
                "backend": "zero-shot-image",
                "version": "large-commit",
                **IMAGE,
            },
        ],
        "aliases": [],
    }


def test_health_probes_each_distinct_backend_once(rclient, fleet):
    resp = rclient.get("/v1/health")
    assert resp.status_code == 200
    probed = sorted(f"{r.url.scheme}://{r.url.host}:{r.url.port}" for r in fleet.requests)
    assert probed == sorted([LAYA_A, LAYA_B, IMG_A, IMG_B])
    assert all(r.url.path == "/health" and r.method == "GET" for r in fleet.requests)
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {
        "laya-english",
        "laya-multilingual",
        "laya-next",
        "siglip2-base-patch16-224",
        "siglip2-large",
    }
    assert all(m["status"] == "ok" for m in body["models"].values())


def test_health_degraded_reports_per_model_status(rclient, fleet):
    fleet.down = {LAYA_B, IMG_A}
    resp = rclient.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["backends"]["text"]["status"] == "ok"  # default text model is on LAYA_A
    assert body["backends"]["image"] == {"status": "down"}  # default image model is on IMG_A
    status = {k: v["status"] for k, v in body["models"].items()}
    assert status == {
        "laya-english": "ok",
        "laya-multilingual": "ok",
        "laya-next": "down",
        "siglip2-base-patch16-224": "down",
        "siglip2-large": "ok",
    }


def test_health_down_when_every_backend_down(rclient, fleet):
    fleet.down = {LAYA_A, LAYA_B, IMG_A, IMG_B}
    resp = rclient.get("/v1/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "down"


def test_healthz_is_public_and_calls_no_backend(rclient, fleet):
    resp = rclient.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert fleet.requests == []


def test_healthz_ok_even_when_backends_down(rclient, fleet):
    fleet.down = {LAYA_A, LAYA_B, IMG_A, IMG_B}
    assert rclient.get("/healthz").status_code == 200


def test_healthz_not_access_logged(rclient, caplog):
    caplog.set_level(logging.INFO, logger="decision_api.access")
    rclient.get("/healthz")
    assert [r for r in caplog.records if r.name == "decision_api.access"] == []


# ---------------------------------------------------------------- startup


def test_invalid_registry_refuses_to_start(tmp_path):
    bad = tmp_path / "registry.yaml"
    bad.write_text(
        textwrap.dedent(REGISTRY).replace("adapter: zero-shot-image, backend_url", "adapter: nope, backend_url")
    )
    with pytest.raises(RegistryError, match="unknown adapter 'nope'"):
        create_app(Settings(api_keys=(API_KEY,), registry_file=str(bad)))


def test_env_registry_file_is_used(registry_file, monkeypatch, fleet):
    monkeypatch.setenv("DECISION_API_KEYS", API_KEY)
    monkeypatch.setenv("DECISION_API_REGISTRY_FILE", str(registry_file))
    app = create_app(transport=httpx.MockTransport(fleet.handler))
    with TestClient(app) as c:
        ids = [m["id"] for m in c.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).json()["models"]]
    assert "siglip2-large" in ids
