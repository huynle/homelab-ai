"""Encoder interface, zero-shot scoring, and the real SigLIP2 implementation.

torch and transformers are imported lazily inside `SiglipEncoder` so the rest of the service
(and its tests) can run without them.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol

import numpy as np
from PIL import Image


class Encoder(Protocol):
    model_name: str
    model_version: str
    device: str
    logit_scale: float

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """Return a (len(prompts), dim) array of text embeddings."""

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return a (dim,) array for one RGB image."""


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def softmax_scores(image_emb: np.ndarray, text_embs: np.ndarray, logit_scale: float) -> np.ndarray:
    """Softmax over labels of logit_scale * cosine(image, text).

    SigLIP's logit_bias is a constant shift across labels, so it cancels in the softmax.
    """
    cos = _normalize(text_embs) @ _normalize(np.reshape(image_emb, -1))
    logits = logit_scale * cos
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


class TextEmbeddingCache:
    """Thread-safe LRU of normalized text embeddings keyed by (template, labels)."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._data: OrderedDict[tuple[str, tuple[str, ...]], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


class Classifier:
    """Zero-shot classification over an Encoder, with text-embedding caching."""

    def __init__(self, encoder: Encoder, cache_size: int = 64):
        self.encoder = encoder
        self.cache = TextEmbeddingCache(cache_size)
        # One forward pass at a time: the model is shared and torch inference is not reentrant-safe.
        self._lock = threading.Lock()

    def text_embeddings(self, template: str, labels: tuple[str, ...]) -> np.ndarray:
        key = (template, labels)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        prompts = [template.replace("{}", label) for label in labels]
        with self._lock:
            embs = _normalize(self.encoder.encode_text(prompts))
        self.cache.put(key, embs)
        return embs

    def classify(self, image: Image.Image, labels: list[str], template: str) -> list[tuple[str, float]]:
        labels_t = tuple(labels)
        text = self.text_embeddings(template, labels_t)
        with self._lock:
            img = self.encoder.encode_image(image)
        scores = softmax_scores(img, text, float(self.encoder.logit_scale))
        order = np.argsort(-scores, kind="stable")
        return [(labels_t[i], float(scores[i])) for i in order]


class SiglipEncoder:
    """Real SigLIP2 encoder (fp32). Loads weights from the HF cache (HF_HOME, HF_HUB_OFFLINE)."""

    TEXT_MAX_LENGTH = 64  # SigLIP was trained with text padded to 64 tokens.

    def __init__(self, model_id: str, revision: str, device: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_id.rsplit("/", 1)[-1]
        self.model_version = revision
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        self.model.to(device).eval()
        self.logit_scale = float(self.model.logit_scale.exp().item())

    @staticmethod
    def _features(out):
        # transformers may return a tensor or a ModelOutput with pooler_output.
        return out if hasattr(out, "detach") else out.pooler_output

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=prompts,
            padding="max_length",
            max_length=self.TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            feats = self._features(self.model.get_text_features(**inputs))
        return feats.float().cpu().numpy()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            feats = self._features(self.model.get_image_features(**inputs))
        return feats.float().cpu().numpy()[0]
