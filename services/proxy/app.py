"""Decision API proxy: one contract (api/decision-api.yaml) in front of the model backends in the registry.

Run with ``python app.py`` or ``uvicorn app:create_app --factory``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from adapters import ADAPTERS, Call, Result
from config import Settings
from errors import ApiError, error_response
from models import (
    MAX_IMAGE_B64_CHARS,
    MAX_IMAGE_BYTES,
    MAX_LABELS,
    MAX_QUESTIONS,
    MAX_STATE_BYTES,
    ImageClassifyRequest,
    TextDecisionRequest,
)
from registry import IMAGE_TASK, TEXT_TASK, Registry, load_registry, registry_from_settings

log = logging.getLogger("decision_api")
access_log = logging.getLogger("decision_api.access")

TEXT_BODY_LIMIT = 64 * 1024
IMAGE_BODY_LIMIT = 12 * 1024 * 1024
LIVENESS_PATH = "/healthz"
PUBLIC_PATHS = {"/v1/health", LIVENESS_PATH}


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


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    registry: Registry | None = None,
) -> FastAPI:
    settings = Settings.from_env() if settings is None else settings
    if registry is None:
        registry = load_registry(settings.registry_file) if settings.registry_file else registry_from_settings(settings)
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
        version="1.1.0",
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
        if request.url.path == LIVENESS_PATH:  # kubelet probes: no auth, no access-log noise
            return await call_next(request)
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

    async def run(request: Request, task: str, data: dict, req) -> JSONResponse:
        entry, explicit = registry.resolve(task, req.model)
        call = Call(client(request), entry, explicit, registry)
        result: Result = await ADAPTERS[entry.adapter].run(call, data, req)
        return JSONResponse(
            {
                "model": result.model,
                "model_version": result.model_version,
                "latency_ms": _ms(request.state.start),
                "results": result.results,
                "meta": result.meta,
            }
        )

    @app.post("/v1/decide/text")
    async def decide_text(request: Request):
        data = _require_object(await read_json(request, TEXT_BODY_LIMIT))
        questions = data.get("questions")
        if isinstance(questions, dict) and len(questions) > MAX_QUESTIONS:
            raise ApiError("invalid_request", f"questions: at most {MAX_QUESTIONS} questions are allowed")
        if "state" in data and (size := _state_bytes(data["state"])) > MAX_STATE_BYTES:
            raise ApiError("payload_too_large", f"state is {size} bytes; limit is {MAX_STATE_BYTES}")
        req = _validate(TextDecisionRequest, data)
        return await run(request, TEXT_TASK, data, req)

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
            # validate=True rejects non-alphabet characters; works on Python 3.10 (a2b_base64 strict_mode is 3.11+)
            base64.b64decode(req.image.base64, validate=True)
        except (binascii.Error, ValueError):
            raise ApiError("invalid_request", "image.base64: not valid standard base64") from None
        return await run(request, IMAGE_TASK, data, req)

    backend_urls = registry.backend_urls()

    @app.get("/v1/health")
    async def health(request: Request):
        http = client(request)
        probes = await asyncio.gather(*(probe(http, url, settings.health_timeout_s) for url in backend_urls))
        by_url = dict(zip(backend_urls, probes, strict=True))
        up = [p["status"] == "ok" for p in probes]
        status = "ok" if all(up) else "degraded" if any(up) else "down"
        return JSONResponse(
            {
                "status": status,
                "backends": {
                    "text": by_url[registry.default(TEXT_TASK).backend_url],
                    "image": by_url[registry.default(IMAGE_TASK).backend_url],
                },
                "models": {m.id: by_url[m.backend_url] for m in registry.models},
            },
            status_code=503 if status == "down" else 200,
        )

    @app.get(LIVENESS_PATH)
    async def healthz():
        return {"status": "ok"}

    models = [{"id": m.id, "task": m.task, "backend": m.backend, "version": m.version} for m in registry.models]

    @app.get("/v1/models")
    async def list_models():
        return {"models": models}

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = Settings.from_env()
        app = create_app(settings)
    except (RuntimeError, ValueError) as exc:  # bad keys/registry/env: one clear line, no traceback
        raise SystemExit(f"decision-api: refusing to start: {exc}") from None
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        access_log=False,  # our own access log omits query strings and bodies
        server_header=False,
    )


if __name__ == "__main__":
    main()
