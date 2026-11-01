"""Shared fixtures: mocked backends (httpx.MockTransport) and real Laya captures."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings

REPO_ROOT = Path(__file__).resolve().parents[3]
API_DIR = REPO_ROOT / "api"
FIXTURES = API_DIR / "fixtures"

TEXT_URL = "http://text-backend.test:8000"
IMAGE_URL = "http://image-backend.test:8001"
API_KEY = "test-key-0123456789abcdef"
OTHER_KEY = "second-key-fedcba9876543210"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def laya_predict() -> dict:
    return load_fixture("laya-predict.json")


@pytest.fixture
def laya_request() -> dict:
    return load_fixture("laya-predict.request.json")


@pytest.fixture
def siglip_response() -> dict:
    """What the image backend returns: built from the real SigLIP2 dog capture."""
    dog = load_fixture("siglip-dog-results.json")
    top3 = dog["devices"]["cuda"]["inputs"]["full"]["general"]["top3"]
    return {
        "model": "siglip2-base-patch16-224",
        "model_version": dog["commit"],
        "latency_ms": 64.5,
        "results": [{"label": r["label"], "score": r["prob"]} for r in top3],
    }


@pytest.fixture
def image_request() -> dict:
    dog = load_fixture("siglip-dog-results.json")
    return {
        "image": {"base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/", "media_type": "image/jpeg"},
        "candidate_labels": dog["families"]["general"],
        "hypothesis_template": "a photo of a {}",
        "top_k": 3,
    }


class Backends:
    """Programmable fake text + image backends. Records every request it sees."""

    def __init__(self, laya_predict: dict, siglip_response: dict):
        self.requests: list[httpx.Request] = []
        self.text: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json=copy.deepcopy(laya_predict)
        )
        self.image: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json=copy.deepcopy(siglip_response)
        )
        self.text_health: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json=load_fixture("laya-health.json")
        )
        self.image_health: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json={"status": "ok"}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        path = request.url.path
        if base == TEXT_URL:
            return self.text_health(request) if path == "/health" else self.text(request)
        if base == IMAGE_URL:
            return self.image_health(request) if path == "/health" else self.image(request)
        raise AssertionError(f"unexpected backend url {request.url}")


@pytest.fixture
def backends(laya_predict, siglip_response) -> Backends:
    return Backends(laya_predict, siglip_response)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_keys=(API_KEY, OTHER_KEY),
        text_backend_url=TEXT_URL,
        image_backend_url=IMAGE_URL,
    )


@pytest.fixture
def client(settings, backends):
    app = create_app(settings, transport=httpx.MockTransport(backends.handler))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}
