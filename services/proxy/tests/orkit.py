"""Helpers for the OpenRouter tests: a registry with an openrouter entry and aliases, and a fake OpenRouter."""

from __future__ import annotations

import copy
import json
import textwrap
from collections.abc import Callable
from pathlib import Path

import httpx

from conftest import IMAGE_URL, TEXT_URL

OPENROUTER_URL = "https://openrouter.ai/api"
OPENROUTER_KEY = "sk-or-v1-test-0000000000000000000000000000000000000000"

REGISTRY = f"""
defaults: {{text: laya-english, image: siglip2-base-patch16-224}}
models:
  - id: laya-english
    task: text-decision
    adapter: laya
    backend_url: {TEXT_URL}
    backend_model: english
    version: laya-serve-0.4.20
  - id: laya-multilingual
    task: text-decision
    adapter: laya
    backend_url: {TEXT_URL}
    backend_model: multilingual
    version: laya-serve-0.4.20
  - id: siglip2-base-patch16-224
    task: image-zero-shot
    adapter: zero-shot-image
    backend: siglip2
    backend_url: {IMAGE_URL}
    version: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
  - id: jev
    task: decisions
    adapter: openrouter
    backend_url: {OPENROUTER_URL}
    backend_model: typesafe/jev-1.14
    capabilities: [text, image, box]
    version: jev-1.14
    timeout_s: 60
aliases:
  auto: {{text: laya-english, image: jev}}
  vision: siglip2-base-patch16-224
"""

JEV_RESPONSE = {
    "answers": {
        "is_bug": {"type": "noul", "noul": 0.96},
        "team": {
            "type": "choice",
            "choice": "payments",
            "confidence": 0.75,
            "probabilities": {"account": 0, "frontend": 0.16, "payments": 0.84},
        },
    },
    "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
    "model": "typesafe/jev-1.14-20260917",
    "provider": "TypeSafe",
    "usage": {"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70},
}


def write_registry(tmp_path: Path, text: str = REGISTRY) -> str:
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(text))
    return str(path)


def write_key(tmp_path: Path, key: str = OPENROUTER_KEY) -> str:
    path = tmp_path / "openrouter-key"
    path.write_text(key + "\n")
    path.chmod(0o600)
    return str(path)


class FakeOpenRouter:
    """Answers POST /api/alpha/decisions; records requests. ``reply`` can be swapped per test."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.reply: Callable[[httpx.Request], httpx.Response] = lambda req: httpx.Response(
            200, json=copy.deepcopy(JEV_RESPONSE)
        )

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]

    def wrap(self, fallback: Callable[[httpx.Request], httpx.Response]):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "openrouter.ai":
                self.requests.append(request)
                return self.reply(request)
            return fallback(request)

        return handler
