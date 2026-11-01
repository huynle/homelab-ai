"""Decision API proxy: one contract (api/decision-api.yaml) in front of Laya (text) and SigLIP2 (image).

Run with ``python app.py`` or ``uvicorn app:create_app --factory``.
"""

from __future__ import annotations

import asyncio
import binascii
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config import Settings
from errors import ApiError, error_response
from models import (
    DEFAULT_TEMPLATE,
    MAX_IMAGE_B64_CHARS,
    MAX_IMAGE_BYTES,
    MAX_LABELS,
    MAX_QUESTIONS,
    MAX_STATE_BYTES,
    ImageClassifyRequest,
    TextDecisionRequest,
)

log = logging.getLogger("decision_api")
access_log = logging.getLogger("decision_api.access")

TEXT_BODY_LIMIT = 64 * 1024
IMAGE_BODY_LIMIT = 12 * 1024 * 1024
PUBLIC_PATHS = {"/v1/health"}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def key_id(key: str) -> str:
    """Non-reversible short id for logs."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def match_key(header: str | None, keys: tuple[str, ...]) -> str | None:
    """Return the configured key matching an ``Authorization: Bearer <key>`` header, else None."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    presented = token.encode()
    found = None
    for key in keys:  # compare against every key; no early exit
        if hmac.compare_digest(presented, key.encode()):
            found = key
    return found


# ---------------------------------------------------------------- request parsing


async def read_json(request: Request, limit: int) -> Any:
    """Read the body with a hard byte cap (before any parsing), then decode JSON."""
    too_large = ApiError("payload_too_large", f"request body exceeds {limit} bytes")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise too_large
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > limit:
            raise too_large
    try:
        return json.loads(buf)
    except (ValueError, UnicodeDecodeError):
        raise ApiError("invalid_request", "request body is not valid JSON") from None


def _require_object(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ApiError("invalid_request", "request body must be a JSON object")
    return data


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_input=False, include_url=False)[:5]:
        loc = ".".join(str(p) for p in err["loc"]) or "body"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _validate(model, data: dict):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ApiError("invalid_request", _validation_message(exc)) from None


def _state_bytes(state: Any) -> int:
    if isinstance(state, str):
        return len(state.encode("utf-8"))
    return len(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _decoded_b64_len(s: str) -> int:
    return len(s) * 3 // 4 - (2 if s.endswith("==") else 1 if s.endswith("=") else 0)


# ---------------------------------------------------------------- backend calls


def _backend_message(resp: httpx.Response) -> str:
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
    return data


async def probe(client: httpx.AsyncClient, url: str, timeout: float) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.get(f"{url}/health", timeout=timeout)
    except httpx.HTTPError:
        return {"status": "down"}
    if resp.status_code != 200:
        return {"status": "down"}
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if isinstance(data, dict) and data.get("status", "ok") != "ok":
        return {"status": "down"}
    return {"status": "ok", "latency_ms": _ms(start)}


# ---------------------------------------------------------------- app


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = Settings.from_env() if settings is None else settings
    if not settings.api_keys and not settings.allow_no_auth:
        raise RuntimeError(
            "no API keys configured: set DECISION_API_KEYS_FILE or DECISION_API_KEYS "
            "(or DECISION_API_ALLOW_NO_AUTH=1 for tests only)"
        )
    auth_required = bool(settings.api_keys)
    if not auth_required:
        log.warning("DECISION_API_ALLOW_NO_AUTH=1: serving without authentication")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False) as client:
            app.state.http = client
            yield

    app = FastAPI(
        title="Homelab Decision API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(exc, _ms(request.state.start))

    @app.middleware("http")
    async def access_and_auth(request: Request, call_next):
        start = request.state.start = time.perf_counter()
        kid = None
        try:
            if auth_required and request.url.path not in PUBLIC_PATHS:
                key = match_key(request.headers.get("authorization"), settings.api_keys)
                if key is None:
                    response = error_response(ApiError("unauthorized", "missing or invalid API key"), _ms(start))
                else:
                    kid = key_id(key)
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except Exception as exc:  # never leak internals to the client
            log.error("unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
            response = error_response(ApiError("internal_error", "unexpected error"), _ms(start))
        access_log.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": _ms(start),
                    "key_id": kid,
                }
            )
        )
        return response

    def client(request: Request) -> httpx.AsyncClient:
        return request.app.state.http

    @app.post("/v1/decide/text")
    async def decide_text(request: Request):
        data = _require_object(await read_json(request, TEXT_BODY_LIMIT))
        questions = data.get("questions")
        if isinstance(questions, dict) and len(questions) > MAX_QUESTIONS:
            raise ApiError("invalid_request", f"questions: at most {MAX_QUESTIONS} questions are allowed")
        if "state" in data and (size := _state_bytes(data["state"])) > MAX_STATE_BYTES:
            raise ApiError("payload_too_large", f"state is {size} bytes; limit is {MAX_STATE_BYTES}")
        req = _validate(TextDecisionRequest, data)

        payload = {"state": data["state"], "questions": data["questions"]}
        if req.model is not None:
            payload["model"] = req.model
        laya = await call_backend(
            client(request), "text", f"{settings.text_backend_url}/predict", payload, settings.text_timeout_s
        )

        answers = laya.get("answers")
        routing = laya.get("routing")
        if not isinstance(answers, dict) or not all(isinstance(a, dict) for a in answers.values()):
            raise ApiError("backend_error", "text backend returned an unexpected response")
        routing = {k: v for k, v in routing.items() if k != "repo"} if isinstance(routing, dict) else {}

        if req.min_confidence is not None:
            for answer in answers.values():
                conf = answer.get("confidence")
                answer["low_confidence"] = not isinstance(conf, (int, float)) or conf < req.min_confidence

        checkpoint = routing.get("model") or req.model
        meta: dict[str, Any] = {"routing": routing}
        if "usage" in laya:
            meta["usage"] = laya["usage"]
        if "model" in laya:
            meta["backend_model"] = laya["model"]
        return JSONResponse(
            {
                "model": f"laya-{checkpoint}" if checkpoint else "laya",
                "model_version": settings.laya_model_version,
                "latency_ms": _ms(request.state.start),
                "results": answers,
                "meta": meta,
            }
        )

    @app.post("/v1/classify/image")
    async def classify_image(request: Request):
        data = _require_object(await read_json(request, IMAGE_BODY_LIMIT))
        image = data.get("image")
        b64 = image.get("base64") if isinstance(image, dict) else None
        if isinstance(b64, str):
            size = _decoded_b64_len(b64)
            if len(b64) > MAX_IMAGE_B64_CHARS or size > MAX_IMAGE_BYTES:
                raise ApiError("payload_too_large", f"decoded image is {size} bytes; limit is {MAX_IMAGE_BYTES}")
        labels = data.get("candidate_labels")
        if isinstance(labels, list) and len(labels) > MAX_LABELS:
            raise ApiError("invalid_request", f"candidate_labels: at most {MAX_LABELS} labels are allowed")
        req = _validate(ImageClassifyRequest, data)
        try:
            binascii.a2b_base64(req.image.base64, strict_mode=True)
        except (binascii.Error, ValueError):
            raise ApiError("invalid_request", "image.base64: not valid standard base64") from None

        resolved = {"model": settings.image_model_id, "model_version": settings.image_model_version}
        backend = await call_backend(
            client(request),
            "image",
            f"{settings.image_backend_url}/classify",
            data,
            settings.image_timeout_s,
            **resolved,
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
        if "latency_ms" in backend:
            meta["backend_latency_ms"] = backend["latency_ms"]
        if "model" in backend:
            meta["backend_model"] = backend["model"]
        return JSONResponse(
            {
                "model": settings.image_model_id,
                "model_version": backend.get("model_version") or settings.image_model_version,
                "latency_ms": _ms(request.state.start),
                "results": results,
                "meta": meta,
            }
        )

    @app.get("/v1/health")
    async def health(request: Request):
        http = client(request)
        text, image = await asyncio.gather(
            probe(http, settings.text_backend_url, settings.health_timeout_s),
            probe(http, settings.image_backend_url, settings.health_timeout_s),
        )
        up = [b["status"] == "ok" for b in (text, image)]
        status = "ok" if all(up) else "degraded" if any(up) else "down"
        return JSONResponse(
            {"status": status, "backends": {"text": text, "image": image}},
            status_code=503 if status == "down" else 200,
        )

    models = [
        {"id": f"laya-{c}", "task": "text-decision", "backend": "laya", "version": settings.laya_model_version}
        for c in settings.laya_checkpoints
    ] + [
        {
            "id": settings.image_model_id,
            "task": "image-zero-shot",
            "backend": "siglip2",
            "version": settings.image_model_version,
        }
    ]

    @app.get("/v1/models")
    async def list_models():
        return {"models": models}

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        access_log=False,  # our own access log omits query strings and bodies
        server_header=False,
    )


if __name__ == "__main__":
    main()
