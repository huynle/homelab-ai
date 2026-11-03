"""Backend adapters: how one kind of model backend is called and mapped onto the contract envelope.

Each adapter is a small class registered in :data:`ADAPTERS` under the key used by ``adapter:`` in the
registry. Supporting a new backend type means writing one class with a ``task`` and a ``run`` method and
adding it to the dict.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

import httpx

from errors import ApiError
from models import DEFAULT_PAD_TO_SQUARE, DEFAULT_TEMPLATE

if TYPE_CHECKING:
    from registry import ModelEntry, Registry


@dataclass
class Result:
    """What an adapter hands back; the route adds ``latency_ms`` and renders the envelope."""

    model: str
    model_version: str
    results: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Call:
    http: httpx.AsyncClient
    entry: ModelEntry
    explicit: bool  # the client named this model; False when it came from the registry defaults
    registry: Registry


# ---------------------------------------------------------------- backend HTTP

# Absolute filesystem paths (POSIX with at least two segments, or Windows drive paths) in backend messages.
_PATH = re.compile(r"(?<![\w:/.])/(?:[^\s/]+/)+[^\s]*|\b[A-Za-z]:\\[^\s]*")


def _redact(message: str) -> str:
    return _PATH.sub("[path]", message)


def _finite(value: Any) -> bool:
    """Every number in a decoded backend JSON value is finite (Python's json accepts NaN and Infinity)."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def _backend_message(resp: httpx.Response) -> str:
    return _redact(_raw_backend_message(resp))


def _raw_backend_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text.strip()[:500] or f"HTTP {resp.status_code}"
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            msgs = []
            for item in detail[:5]:
                if isinstance(item, dict):
                    loc = ".".join(str(p) for p in item.get("loc", []))
                    msgs.append(f"{loc}: {item.get('msg', '')}".strip(": "))
                else:
                    msgs.append(str(item))
            return "; ".join(msgs)
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
        if isinstance(data.get("message"), str):
            return data["message"]
    return json.dumps(data)[:500]


async def call_backend(
    client: httpx.AsyncClient, name: str, url: str, payload: dict, timeout: float, **err_fields
) -> dict:
    """POST JSON to a backend and map every failure onto a contract error."""
    netloc = urlsplit(url).netloc
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise ApiError("backend_unavailable", f"{name} backend unreachable at {netloc}", **err_fields) from None
    except httpx.TimeoutException:
        raise ApiError(
            "backend_timeout", f"{name} backend did not answer within {int(timeout * 1000)} ms", **err_fields
        ) from None
    except httpx.TransportError as exc:
        raise ApiError(
            "backend_error", f"{name} backend connection failed ({type(exc).__name__})", **err_fields
        ) from None

    if 400 <= resp.status_code < 500:
        raise ApiError(
            "invalid_request", f"{name} backend rejected the request: {_backend_message(resp)}", **err_fields
        )
    if resp.status_code != 200:
        raise ApiError("backend_error", f"{name} backend returned HTTP {resp.status_code}", **err_fields)
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise ApiError("backend_error", f"{name} backend returned an unparseable response", **err_fields)
    if not _finite(data):
        raise ApiError("backend_error", f"{name} backend returned a non-finite number", **err_fields)
    return data


# ---------------------------------------------------------------- adapters


class Adapter:
    task: ClassVar[str]

    async def run(self, call: Call, data: dict, req: Any) -> Result:  # pragma: no cover - interface
        raise NotImplementedError


class LayaAdapter(Adapter):
    """Laya ``POST /predict``: typed text decisions."""

    task = "text-decision"

    async def run(self, call: Call, data: dict, req: Any) -> Result:
        entry = call.entry
        payload = {"state": data["state"], "questions": data["questions"]}
        # A pinned checkpoint is forwarded only when the client chose the model. Requests that fall back to
        # the default keep Laya's own script/language routing (the contract's "omitted model" behaviour).
        pinned = call.explicit and entry.backend_model is not None
        if pinned:
            payload["model"] = entry.backend_model
        laya = await call_backend(call.http, "text", f"{entry.backend_url}/predict", payload, entry.timeout_s)

        answers = laya.get("answers")
        routing = laya.get("routing")
        if not isinstance(answers, dict) or not all(isinstance(a, dict) for a in answers.values()):
            raise ApiError("backend_error", "text backend returned an unexpected response")
        routing = {k: v for k, v in routing.items() if k != "repo"} if isinstance(routing, dict) else {}

        if req.min_confidence is not None:
            for answer in answers.values():
                conf = answer.get("confidence")
                answer["low_confidence"] = not isinstance(conf, (int, float)) or conf < req.min_confidence

        served = entry
        checkpoint = routing.get("model")
        if not pinned and isinstance(checkpoint, str):
            routed = call.registry.find(f"laya-{checkpoint}")
            if routed is not None and routed.adapter == "laya" and routed.backend_url == entry.backend_url:
                served = routed

        meta: dict[str, Any] = {"routing": routing}
        if "usage" in laya:
            meta["usage"] = laya["usage"]
        if "model" in laya:
            meta["backend_model"] = laya["model"]
        return Result(served.id, served.version, answers, meta)


class ZeroShotImageAdapter(Adapter):
    """Zero-shot image classification service ``POST /classify`` (services/image, SigLIP2)."""

    task = "image-zero-shot"

    async def run(self, call: Call, data: dict, req: Any) -> Result:
        entry = call.entry
        resolved = {"model": entry.id, "model_version": entry.version}
        payload = {k: v for k, v in data.items() if k != "model"}  # the registry id stays in the proxy
        backend = await call_backend(
            call.http, "image", f"{entry.backend_url}/classify", payload, entry.timeout_s, **resolved
        )
        results = backend.get("results")
        if not isinstance(results, list) or not all(
            isinstance(r, dict) and "label" in r and "score" in r for r in results
        ):
            raise ApiError("backend_error", "image backend returned an unexpected response", **resolved)

        meta: dict[str, Any] = dict(backend["meta"]) if isinstance(backend.get("meta"), dict) else {}
        if "device" in backend:
            meta.setdefault("device", backend["device"])
        meta["hypothesis_template"] = req.hypothesis_template or DEFAULT_TEMPLATE
        meta["pad_to_square"] = DEFAULT_PAD_TO_SQUARE if req.pad_to_square is None else req.pad_to_square
        if "latency_ms" in backend:
            meta["backend_latency_ms"] = backend["latency_ms"]
        if "model" in backend:
            meta["backend_model"] = backend["model"]
        version = backend.get("model_version")
        return Result(entry.id, version if isinstance(version, str) and version else entry.version, results, meta)


ADAPTERS: dict[str, Adapter] = {
    "laya": LayaAdapter(),
    "zero-shot-image": ZeroShotImageAdapter(),
}
