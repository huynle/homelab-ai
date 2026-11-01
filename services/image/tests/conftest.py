"""Test fixtures: a deterministic fake encoder so tests need no GPU, weights, or network."""

from __future__ import annotations

import base64
import io
import math
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import Settings, create_app

DIM = 256


def _axis(text: str) -> int:
    # Axis 0 is reserved for the image; every prompt gets its own (stable) orthogonal axis.
    return 1 + zlib.crc32(text.encode()) % (DIM - 1)


class FakeEncoder:
    """Encoder whose cosine(image, prompt) is chosen per label via `affinity`.

    A prompt whose text contains a key of `affinity` gets that cosine with the image; any other
    prompt gets 0. Calls are recorded so tests can check caching and the image passed in.
    """

    model_name = "siglip2-base-patch16-224"
    model_version = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    device = "cpu"
    logit_scale = 10.0

    def __init__(self, affinity: dict[str, float] | None = None, fail: bool = False):
        self.affinity = affinity or {"dog": 0.9, "cat": 0.5}
        self.fail = fail
        self.text_calls: list[list[str]] = []
        self.image_sizes: list[tuple[int, int]] = []

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        self.text_calls.append(list(prompts))
        out = np.zeros((len(prompts), DIM), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            cos = next((c for k, c in self.affinity.items() if k in prompt), 0.0)
            out[i, 0] = cos
            out[i, _axis(prompt)] = math.sqrt(1.0 - cos * cos)
        return out

    def encode_image(self, image: Image.Image) -> np.ndarray:
        if self.fail:
            raise RuntimeError("boom")
        self.image_sizes.append(image.size)
        vec = np.zeros(DIM, dtype=np.float32)
        vec[0] = 1.0
        return vec


def png_b64(size: tuple[int, int] = (32, 32), color: str = "red") -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def client(encoder, settings):
    app = create_app(settings=settings, encoder=encoder)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
