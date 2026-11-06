"""Contract error codes and the ErrorEnvelope response."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

STATUS = {
    "invalid_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "payload_too_large": 413,
    "rate_limited": 429,
    "internal_error": 500,
    "backend_error": 502,
    "backend_unavailable": 503,
    "backend_timeout": 504,
}

# OpenAI-style `error.type`, so OpenAI SDK clients (POST /v1/decisions) classify our errors as they do theirs.
TYPE = {
    "invalid_request": "invalid_request_error",
    "payload_too_large": "invalid_request_error",
    "unauthorized": "authentication_error",
    "forbidden": "permission_error",
    "rate_limited": "rate_limit_error",
}


class ApiError(Exception):
    """Raised anywhere in a request; rendered as an ErrorEnvelope with the code's HTTP status."""

    def __init__(self, code: str, message: str, *, param: str | None = None, **fields: Any):
        assert code in STATUS, code
        super().__init__(message)
        self.code = code
        self.message = message
        self.param = param
        self.fields = {k: v for k, v in fields.items() if v is not None}

    @property
    def status(self) -> int:
        return STATUS[self.code]


def error_response(err: ApiError, latency_ms: float | None = None) -> JSONResponse:
    body: dict[str, Any] = dict(err.fields)
    if latency_ms is not None:
        body.setdefault("latency_ms", latency_ms)
    body["error"] = {
        "code": err.code,
        "message": err.message,
        "type": TYPE.get(err.code, "server_error"),
        "param": err.param,
    }
    headers = {"WWW-Authenticate": 'Bearer realm="decision-api"'} if err.code == "unauthorized" else None
    return JSONResponse(body, status_code=err.status, headers=headers)
