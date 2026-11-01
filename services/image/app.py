"""SigLIP2 zero-shot image classification server.

Internal backend for the decision-api proxy (api/decision-api.yaml, `POST /v1/classify/image`).

    POST /classify  -> {model, model_version, latency_ms, results: [{label, score}]}
    GET  /health    -> {status, model, model_version, device}

Errors use the contract shape {"error": {"code", "message"}}.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from encoder import Classifier, Encoder
from imaging import ImageError, ImageTooLarge, decode_base64, load_image, pad_to_square

__all__ = ["Settings", "create_app", "pad_to_square"]

log = logging.getLogger("image")

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DEFAULT_TEMPLATE = "a photo of a {}"
MAX_LABELS = 64


@dataclass(frozen=True)
class Settings:
    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_REVISION
    device: str = "auto"
    max_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 64_000_000
    text_cache_size: int = 64
    host: str = "127.0.0.1"
    port: int = 8010

    @property
    def model_name(self) -> str:
        return self.model_id.rsplit("/", 1)[-1]

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        d = cls()
        return cls(
            model_id=env.get("IMAGE_MODEL_ID", d.model_id),
            revision=env.get("IMAGE_MODEL_REVISION", d.revision),
            device=env.get("IMAGE_DEVICE", d.device),
            max_bytes=int(env.get("IMAGE_MAX_BYTES", d.max_bytes)),
            max_pixels=int(env.get("IMAGE_MAX_PIXELS", d.max_pixels)),
            text_cache_size=int(env.get("IMAGE_TEXT_CACHE_SIZE", d.text_cache_size)),
            host=env.get("HOST", d.host),
            port=int(env.get("PORT", d.port)),
        )


# ---------- request schema (mirrors ImageClassifyRequest in the contract) ----------

Label = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class ImageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Length limit is enforced in imaging.decode_base64 so oversize maps to 413, not 400.
    base64: str = Field(min_length=4)
    media_type: Optional[Literal["image/jpeg", "image/png"]] = None


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: ImageIn
    candidate_labels: list[Label] = Field(min_length=1, max_length=MAX_LABELS)
    hypothesis_template: str = Field(DEFAULT_TEMPLATE, max_length=512, pattern=r"\{\}")
    top_k: Optional[int] = Field(None, ge=1, le=MAX_LABELS)
    pad_to_square: bool = True

    @field_validator("candidate_labels")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("candidate_labels must be unique")
        return v


# ---------- errors ----------


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid request"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    msg = first.get("msg", "invalid value")
    return f"{loc}: {msg}" if loc else msg


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        return _error(400, "invalid_request", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        if exc.status_code == 413:
            return _error(413, "payload_too_large", str(exc.detail))
        if 400 <= exc.status_code < 500:
            return _error(exc.status_code, "invalid_request", str(exc.detail))
        return _error(500, "internal_error", "unexpected error")

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.error("unhandled %s", type(exc).__name__)
        return _error(500, "internal_error", "unexpected error")


# ---------- app ----------


def create_app(settings: Optional[Settings] = None, encoder: Optional[Encoder] = None) -> FastAPI:
    """Build the app. Pass `encoder` to inject one (tests); otherwise SigLIP2 loads at startup."""
    settings = settings or Settings.from_env()
    Image.MAX_IMAGE_PIXELS = settings.max_pixels

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        enc = encoder
        if enc is None:
            from encoder import SiglipEncoder

            log.info("loading %s@%s", settings.model_id, settings.revision)
            enc = SiglipEncoder(settings.model_id, settings.revision, settings.device)
            log.info("loaded on %s, logit_scale=%.3f", enc.device, enc.logit_scale)
        app.state.encoder = enc
        app.state.classifier = Classifier(enc, settings.text_cache_size)
        yield

    app = FastAPI(title="siglip2 image classifier", lifespan=lifespan)
    _install_error_handlers(app)

    @app.get("/health")
    def health(request: Request):
        enc = request.app.state.encoder
        return {
            "status": "ok",
            "model": enc.model_name,
            "model_version": enc.model_version,
            "device": enc.device,
        }

    @app.post("/classify")
    def classify(body: ClassifyRequest, request: Request):
        # Sync handler: FastAPI runs it in a threadpool so inference does not block the loop.
        start = time.perf_counter()
        try:
            raw = decode_base64(body.image.base64, settings.max_bytes)
            img = load_image(raw, settings.max_pixels)
        except ImageTooLarge as exc:
            raise ApiError(413, "payload_too_large", str(exc)) from None
        except ImageError as exc:
            raise ApiError(400, "invalid_request", str(exc)) from None
        if body.pad_to_square:
            img = pad_to_square(img)

        enc = request.app.state.encoder
        try:
            ranked = request.app.state.classifier.classify(
                img, body.candidate_labels, body.hypothesis_template
            )
        except Exception as exc:
            log.error("inference failed: %s", type(exc).__name__)
            raise ApiError(500, "internal_error", "image inference failed") from None
        if body.top_k is not None:
            ranked = ranked[: body.top_k]
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(
            "classify labels=%d size=%dx%d latency_ms=%.1f",
            len(body.candidate_labels), img.size[0], img.size[1], latency_ms,
        )
        return {
            "model": enc.model_name,
            "model_version": enc.model_version,
            "latency_ms": latency_ms,
            "results": [{"label": label, "score": score} for label, score in ranked],
        }

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False)


if __name__ == "__main__":
    main()
