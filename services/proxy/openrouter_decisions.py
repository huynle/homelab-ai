"""OpenRouter Decisions API (``POST /api/alpha/decisions``) on top of the registry.

Validates OpenRouter's ``DecisionsRequest`` (``model``, ``state``, ``questions`` keyed by id, ``assets``, ...),
works out what the request needs (images, audio, video, box questions, text), and then either forwards it to
OpenRouter unchanged (adapter ``openrouter``, see adapters.py) or translates it for a local backend: Laya
``/predict`` (text, JSON) or SigLIP2 ``/classify`` (one image, choice questions only). Answers come back in
OpenRouter's ``DecisionsResponse`` shape. Errors on ``/api/alpha/*`` use OpenRouter's error envelope.

References: https://openrouter.ai/docs/guides/community/multimodal-decisions and the OpenAPI operation
``createApiAlphaDecisions``.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import secrets
import string
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi.responses import JSONResponse

from errors import ApiError
from models import MAX_IMAGE_B64_CHARS, MAX_IMAGE_BYTES, MAX_LABELS, MAX_QUESTIONS, MAX_STATE_BYTES

PATH_PREFIX = "/api/alpha/"
BODY_LIMIT = 16 * 1024 * 1024  # images travel inline (data URLs or assets)
MAX_DECISION_QUESTIONS = 400  # the proxy's own cap, for every model
MAX_ID_CHARS = 256  # session_id and user (OpenRouter's limits)
MAX_ASSET_KEY_CHARS = 256
LOCAL_CHOICES = (2, MAX_LABELS)  # Laya criteria and SigLIP2 labels
LOCAL_LEVELS = (2, 16)  # Laya score criteria
MAX_LOCAL_INSTRUCTIONS = 2048  # Laya
MAX_LABEL_CHARS = 256  # SigLIP2
LAYA_CHUNK = MAX_QUESTIONS  # Laya answers at most 32 questions per call
FALLBACK_INSTRUCTIONS = "Answer this question about the state."  # Laya needs instructions; OpenRouter does not
LOCAL_PROVIDER = "homelab"

REQUEST_FIELDS = frozenset({"model", "state", "questions", "assets", "provider", "session_id", "trace", "user"})
FORWARDED_FIELDS = ("state", "questions", "assets", "provider")  # session_id, trace and user stay in the proxy
QUESTION_FIELDS = {
    "noul": frozenset({"type", "instructions", "criteria"}),
    "choice": frozenset({"type", "instructions", "criteria"}),
    "score": frozenset({"type", "instructions", "criteria"}),
    "box": frozenset({"type", "instructions"}),
}
ASSET_FIELDS = frozenset({"type", "format", "data_base64"})
ASSET_FORMATS = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
MEDIA_PARTS = {"image_url": "image", "input_audio": "audio", "video_url": "video"}
CAPABILITY_NAMES = {"image": "images", "audio": "audio", "video": "video", "box": "box questions", "text": "text"}
QID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
B64 = re.compile(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")
DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|jpg|webp));base64,(.*)$", re.DOTALL)
LOCAL_IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})
ID_ALPHABET = string.ascii_letters + string.digits


def bad(param: str, message: str) -> ApiError:
    return ApiError("invalid_request", f"{param}: {message}", param=param)


def is_alpha_path(path: str) -> bool:
    return path.startswith(PATH_PREFIX)


def error_response(err: ApiError) -> JSONResponse:
    """OpenRouter's error envelope: ``error.code`` is the HTTP status; our code and param go in ``metadata``."""
    body = {
        "error": {
            "code": err.status,
            "message": err.message,
            "metadata": {"code": err.code, "param": err.param},
        }
    }
    headers = {"WWW-Authenticate": 'Bearer realm="decision-api"'} if err.code == "unauthorized" else None
    return JSONResponse(body, status_code=err.status, headers=headers)


@dataclass
class Image:
    base64: str
    media_type: str
    param: str


@dataclass
class Plan:
    model: str
    raw: dict  # the validated request body (what OpenRouter gets, minus proxy-only fields)
    questions: dict[str, dict]
    assets: dict[str, dict] = field(default_factory=dict)
    needs: set[str] = field(default_factory=set)  # image, audio, video, box, text
    state_image: Image | None = None  # the single image that is the whole state (SigLIP2), if any
    images: int = 0  # image_url parts anywhere in state, instructions and criteria

    @property
    def media(self) -> bool:
        """An alias picks its image target: media parts, assets or box questions."""
        return bool(self.needs & {"image", "audio", "video", "box"}) or bool(self.assets)


# ---------------------------------------------------------------- parsing


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _guidance(value: Any, param: str, *, nullable: bool) -> None:
    """A plain string, or a JSON object or array of structured guidance (or null where the schema allows)."""
    if value is None and nullable:
        return
    if not isinstance(value, (str, dict, list)):
        allowed = "a string, an object, an array" + (" or null" if nullable else "")
        raise bad(param, f"must be {allowed}")


def _data_url(url: Any, param: str) -> Image:
    if not isinstance(url, str):
        raise bad(param, "must be a string")
    if url.startswith(("http://", "https://")):
        raise bad(param, "hosted image URLs are not supported; send the image as a base64 data URL")
    m = DATA_URL.match(url)
    if m is None:
        raise bad(param, "must be a base64 data URL of a PNG, JPEG or WebP image (data:image/png;base64,...)")
    media_type, b64 = m.group(1), m.group(2)
    if not b64:
        raise bad(param, "the data URL holds no image data")
    try:
        base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise bad(param, "not valid standard base64") from None
    return Image(b64, "image/jpeg" if media_type == "image/jpg" else media_type, param)


class _Scan:
    """Walks state, instructions and criteria: media parts, the strings (for asset references) and text."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.strings: list[str] = []
        self.last_image: Image | None = None

    def part(self, value: dict, param: str) -> bool:
        """Validate a media part; False when ``value`` is not one."""
        kind = value.get("type")
        if not isinstance(kind, str) or kind not in MEDIA_PARTS:
            return False
        self.plan.needs.add(MEDIA_PARTS[kind])
        if kind not in value:
            raise bad(param, f"a {kind} part needs a {kind} field")
        inner = value[kind]
        if kind == "image_url":
            url = inner.get("url") if isinstance(inner, dict) else inner
            p = f"{param}.image_url.url" if isinstance(inner, dict) else f"{param}.image_url"
            self.last_image = _data_url(url, p)
            self.plan.images += 1
        elif kind == "video_url":
            url = inner.get("url") if isinstance(inner, dict) else inner
            if not isinstance(url, str) or not url:
                raise bad(f"{param}.video_url", "must hold a url")
        elif not isinstance(inner, dict):
            raise bad(f"{param}.input_audio", "must be an object")
        return True

    def walk(self, value: Any, param: str) -> None:
        if isinstance(value, str):
            self.strings.append(value)
        elif isinstance(value, dict):
            if self.part(value, param):
                return
            for k, v in value.items():
                self.walk(v, f"{param}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                self.walk(v, f"{param}[{i}]")


def _asset(key: Any, asset: Any) -> None:
    p = f"assets.{key}"
    if not isinstance(key, str) or not 1 <= len(key) <= MAX_ASSET_KEY_CHARS:
        raise bad("assets", f"asset keys must be 1 to {MAX_ASSET_KEY_CHARS} characters")
    if not isinstance(asset, dict):
        raise bad(p, "must be an object with format and data_base64")
    extra = sorted(set(asset) - ASSET_FIELDS)
    if extra:
        raise bad(f"{p}.{extra[0]}", "unknown field for an asset")
    if "type" in asset and asset["type"] != "image":
        raise bad(f"{p}.type", "must be image")
    if asset.get("format") not in ASSET_FORMATS:
        raise bad(f"{p}.format", "must be png, jpg or jpeg")
    data = asset.get("data_base64")
    if not isinstance(data, str) or not data or not B64.match(data):
        raise bad(f"{p}.data_base64", "must be standard base64 (no data: prefix)")


def _question(qid: Any, q: Any, scan: _Scan) -> None:
    if not isinstance(qid, str) or not QID.match(qid):
        raise bad("questions", "question ids must match ^[A-Za-z0-9_.-]{1,64}$")
    p = f"questions.{qid}"
    if not isinstance(q, dict):
        raise bad(p, "must be an object")
    kind = q.get("type")
    if not isinstance(kind, str) or kind not in QUESTION_FIELDS:
        raise bad(f"{p}.type", f"{kind!r} is not a question type (noul, choice, score, box)")
    extra = sorted(set(q) - QUESTION_FIELDS[kind])
    if extra:
        raise bad(f"{p}.{extra[0]}", f"unknown field for a {kind} question")
    if kind == "box":
        if "instructions" not in q:
            raise bad(f"{p}.instructions", "is required for a box question")
        _guidance(q["instructions"], f"{p}.instructions", nullable=False)
        scan.plan.needs.add("box")
    else:
        _guidance(q.get("instructions"), f"{p}.instructions", nullable=True)
    scan.walk(q.get("instructions"), f"{p}.instructions")

    criteria = q.get("criteria")
    cp = f"{p}.criteria"
    if kind == "noul" and criteria is not None:
        if not isinstance(criteria, dict):
            raise bad(cp, "must be an object with optional true and false")
        for key, value in criteria.items():
            if key not in ("true", "false"):
                raise bad(f"{cp}.{key}", "noul criteria accept only true and false")
            _guidance(value, f"{cp}.{key}", nullable=True)
    elif kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise bad(cp, "must be an object of options (option -> description or null)")
        for key, value in criteria.items():
            _guidance(value, f"{cp}.{key}", nullable=True)
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise bad(cp, "must be a non-empty array of levels, lowest first")
        for i, value in enumerate(criteria):
            _guidance(value, f"{cp}[{i}]", nullable=False)
    scan.walk(criteria, cp)


def _state_text(state: Any, assets: dict) -> bool:
    """Whether ``state`` carries anything besides media parts and whole-string asset references."""

    def textual(item: Any) -> bool:
        if isinstance(item, str):
            return item not in assets
        if isinstance(item, dict):
            return item.get("type") not in MEDIA_PARTS
        return True

    if isinstance(state, list):
        return any(textual(item) for item in state)
    return textual(state)


def _whole_state_image(state: Any, plan: Plan) -> Image | None:
    """The image that is the whole state: an image part, an asset key, or the only item of a state array."""
    only, param = state, "state"
    if isinstance(state, list):
        if len(state) != 1:
            return None
        only, param = state[0], "state[0]"
    if isinstance(only, dict) and only.get("type") == "image_url":
        inner = only["image_url"]
        url = inner.get("url") if isinstance(inner, dict) else inner
        return _data_url(url, param)
    if isinstance(only, str) and only in plan.assets:
        asset = plan.assets[only]
        return Image(asset["data_base64"], ASSET_FORMATS[asset["format"]], f"assets.{only}")
    return None


def parse(data: dict) -> Plan:
    extra = sorted(set(data) - REQUEST_FIELDS)
    if extra:
        raise bad(extra[0], "unknown parameter")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise bad("model", "is required (a model id or alias from GET /v1/models)")
    if "state" not in data:
        raise bad("state", "is required")
    state = data["state"]
    if not isinstance(state, (str, dict, list)):
        raise bad("state", "must be a string, an object or an array")
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise bad("questions", "must be a non-empty object of questions keyed by id")
    if len(questions) > MAX_DECISION_QUESTIONS:
        raise bad("questions", f"at most {MAX_DECISION_QUESTIONS} questions are supported")
    for name in ("session_id", "user"):
        if name in data and (not isinstance(data[name], str) or len(data[name]) > MAX_ID_CHARS):
            raise bad(name, f"must be a string of at most {MAX_ID_CHARS} characters")
    if "trace" in data and not isinstance(data["trace"], dict):
        raise bad("trace", "must be an object")
    if "provider" in data and data["provider"] is not None and not isinstance(data["provider"], dict):
        raise bad("provider", "must be an object or null")
    assets = data.get("assets", {})
    if not isinstance(assets, dict):
        raise bad("assets", "must be an object of images keyed by reference")
    for key, asset in assets.items():
        _asset(key, asset)

    plan = Plan(model=model, raw=data, questions=questions, assets=assets)
    scan = _Scan(plan)
    scan.walk(state, "state")
    for qid, q in questions.items():
        _question(qid, q, scan)
    for key in assets:
        if not any(key in s for s in scan.strings):
            raise bad(f"assets.{key}", f"asset {key!r} is not referenced in state, instructions or criteria")
    if assets:
        plan.needs.add("image")
    if _state_text(state, assets):
        plan.needs.add("text")
    plan.state_image = _whole_state_image(state, plan)
    return plan


# ---------------------------------------------------------------- model checks


def check_model(entry, plan: Plan) -> None:
    """400 when the resolved registry entry cannot serve the request (before any backend call)."""
    caps = set(entry.capabilities or ())
    for need in ("image", "audio", "video", "box"):
        if need in plan.needs and need not in caps:
            raise ApiError(
                "invalid_request",
                f"model {entry.id!r} does not accept {CAPABILITY_NAMES[need]}",
                param="model",
            )
    if entry.adapter == "zero-shot-image":
        _check_siglip(entry, plan)
    elif "text" in plan.needs and "text" not in caps:
        raise ApiError("invalid_request", f"model {entry.id!r} does not accept text", param="model")


def _check_siglip(entry, plan: Plan) -> None:
    for qid, q in plan.questions.items():
        if q["type"] != "choice":
            raise bad(
                f"questions.{qid}.type",
                f"model {entry.id!r} (zero-shot image classification) answers only choice questions",
            )
    image = plan.state_image
    if image is None or "text" in plan.needs or plan.images + len(plan.assets) != 1:
        raise ApiError(
            "invalid_request",
            f"model {entry.id!r} needs exactly one image as the whole state (an image_url part, an asset key, "
            "or the only item of a state array) and nothing else",
            param="state",
        )
    if image.media_type not in LOCAL_IMAGE_TYPES:
        raise bad(image.param, f"model {entry.id!r} accepts only PNG or JPEG images")
    decoded = len(image.base64) * 3 // 4 - image.base64[-2:].count("=")
    if len(image.base64) > MAX_IMAGE_B64_CHARS or decoded > MAX_IMAGE_BYTES:
        raise ApiError(
            "payload_too_large", f"decoded image is {decoded} bytes; limit is {MAX_IMAGE_BYTES}", param=image.param
        )
    for qid, q in plan.questions.items():
        labels = list(q["criteria"])
        if not LOCAL_CHOICES[0] <= len(labels) <= LOCAL_CHOICES[1]:
            raise bad(
                f"questions.{qid}.criteria", f"local models need {LOCAL_CHOICES[0]} to {LOCAL_CHOICES[1]} options"
            )
        for label in labels:
            if not 1 <= len(label) <= MAX_LABEL_CHARS:
                raise bad(f"questions.{qid}.criteria", f"image options must be 1 to {MAX_LABEL_CHARS} characters")


# ---------------------------------------------------------------- Laya


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else _compact(value)


def _laya_question(qid: str, q: dict) -> dict:
    p = f"questions.{qid}"
    instructions = _text(q.get("instructions")) or FALLBACK_INSTRUCTIONS
    out: dict[str, Any] = {"type": q["type"]}
    criteria = q.get("criteria")
    if q["type"] == "noul":
        # Laya's noul question has no criteria (api/fixtures): fold them into the instructions.
        lines = [instructions]
        for key, label in (("true", "True if"), ("false", "False if")):
            text = _text((criteria or {}).get(key))
            if text:
                lines.append(f"{label}: {text}")
        instructions = "\n".join(lines)
    elif q["type"] == "choice":
        lo, hi = LOCAL_CHOICES
        if not lo <= len(criteria) <= hi:
            raise bad(f"{p}.criteria", f"local models need {lo} to {hi} choice options")
        out["criteria"] = {k: k if v is None else _text(v) for k, v in criteria.items()}
    else:
        lo, hi = LOCAL_LEVELS
        if not lo <= len(criteria) <= hi:
            raise bad(f"{p}.criteria", f"local models need {lo} to {hi} score levels")
        out["criteria"] = [_text(v) for v in criteria]
    if len(instructions) > MAX_LOCAL_INSTRUCTIONS:
        raise bad(f"{p}.instructions", f"at most {MAX_LOCAL_INSTRUCTIONS} characters for local models")
    out["instructions"] = instructions
    return out


def laya_state(state: Any) -> str | dict:
    if isinstance(state, list):
        state = "\n".join(state) if all(isinstance(s, str) for s in state) else _compact(state)
    if isinstance(state, str):
        if not state.strip():
            raise bad("state", "must not be empty for local models")
        size = len(state.encode("utf-8"))
    else:
        size = len(_compact(state).encode("utf-8"))
    if size > MAX_STATE_BYTES:
        raise ApiError("payload_too_large", f"state is {size} bytes; limit is {MAX_STATE_BYTES}", param="state")
    return state


def laya_payloads(plan: Plan) -> list[dict]:
    """Laya ``/predict`` bodies, at most ``LAYA_CHUNK`` questions each, in question order."""
    state = laya_state(plan.raw["state"])
    questions = {qid: _laya_question(qid, q) for qid, q in plan.questions.items()}
    ids = list(questions)
    return [
        {"state": state, "questions": {qid: questions[qid] for qid in ids[i : i + LAYA_CHUNK]}}
        for i in range(0, len(ids), LAYA_CHUNK)
    ]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ApiError("backend_error", "backend returned an unexpected answer")
    return value


def _from_laya(q: dict, ans: Any) -> dict:
    if not isinstance(ans, dict):
        return {"type": "refusal"}
    if q["type"] == "noul":
        return {"type": "noul", "noul": _number(ans.get("noul"))}
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        raise ApiError("backend_error", "backend returned an unexpected answer")
    confidence = _number(ans.get("confidence"))
    if q["type"] == "choice":
        options = list(q["criteria"])
        if ans.get("choice") not in options:
            raise ApiError("backend_error", "backend returned a choice that was not offered")
        return {
            "type": "choice",
            "choice": ans["choice"],
            "probabilities": {o: _number(probs.get(o, 0.0)) for o in options},
            "confidence": confidence,
        }
    levels = q["criteria"]
    return {
        "type": "score",
        "score": _number(ans.get("score")),
        "probabilities": {str(i): _number(probs.get(str(i), 0.0)) for i in range(len(levels))},
        "legend": {str(i): level for i, level in enumerate(levels)},  # as the client sent it
        "confidence": confidence,
    }


def new_id() -> str:
    """An id shaped like OpenRouter's: ``gen-dec-<unix seconds>-<20 alphanumerics>``."""
    return f"gen-dec-{int(time.time())}-{''.join(secrets.choice(ID_ALPHABET) for _ in range(20))}"


def _response(model: str, answers: dict, input_tokens: int = 0, output_tokens: int = 0) -> dict:
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": 0},
        "id": new_id(),
        "provider": LOCAL_PROVIDER,
    }


def render_laya(plan: Plan, model: str, chunks: list[tuple[dict, dict]]) -> dict:
    """``chunks``: (Laya answers by qid, meta) per Laya call."""
    answers: dict[str, Any] = {}
    tokens_in = tokens_out = 0
    for results, meta in chunks:
        answers.update(results)
        u = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        tokens_in += u["input_tokens"] if isinstance(u.get("input_tokens"), int) else 0
        tokens_out += u["output_tokens"] if isinstance(u.get("output_tokens"), int) else 0
    out = {qid: _from_laya(q, answers.get(qid)) for qid, q in plan.questions.items()}
    return _response(model, out, tokens_in, tokens_out)


# ---------------------------------------------------------------- SigLIP2


def siglip_payloads(plan: Plan) -> list[dict]:
    """One SigLIP2 ``/classify`` body per (choice) question, over its option keys."""
    image = plan.state_image
    assert image is not None
    img = {"base64": image.base64, "media_type": image.media_type}
    return [{"image": img, "candidate_labels": list(q["criteria"])} for q in plan.questions.values()]


def _from_siglip(q: dict, results: Any) -> dict:
    if not isinstance(results, list) or not results:
        raise ApiError("backend_error", "image backend returned no labels")
    if not all(isinstance(r, dict) and isinstance(r.get("label"), str) for r in results):
        raise ApiError("backend_error", "image backend returned an unexpected response")
    scores = {r["label"]: _number(r.get("score")) for r in results}
    options = list(q["criteria"])
    top = max(scores, key=scores.__getitem__)
    if top not in options:
        raise ApiError("backend_error", "image backend returned a label that was not offered")
    return {
        "type": "choice",
        "choice": top,
        "probabilities": {o: scores.get(o, 0.0) for o in options},
        "confidence": scores[top],
    }


def render_siglip(plan: Plan, model: str, results: list[Any]) -> dict:
    answers = {qid: _from_siglip(q, r) for (qid, q), r in zip(plan.questions.items(), results, strict=True)}
    return _response(model, answers)


# ---------------------------------------------------------------- OpenRouter


def upstream_body(plan: Plan, backend_model: str) -> dict:
    """What OpenRouter gets: the client's body with the upstream model id; session_id, trace and user dropped."""
    body: dict[str, Any] = {"model": backend_model}
    for name in FORWARDED_FIELDS:
        if name in plan.raw:
            body[name] = plan.raw[name]
    return body


def render_upstream(model: str, answers: dict, meta: dict) -> dict:
    usage = meta.get("usage")
    out: dict[str, Any] = {
        "model": model,  # the registry id, not OpenRouter's versioned model id
        "answers": answers,
        "usage": usage if isinstance(usage, dict) else {"input_tokens": 0, "output_tokens": 0},
        "id": meta["id"] if isinstance(meta.get("id"), str) else new_id(),
        "provider": meta["provider"] if isinstance(meta.get("provider"), str) else "openrouter",
    }
    if isinstance(meta.get("annotations"), list):
        out["annotations"] = meta["annotations"]
    return out


def usage_cost(body: dict) -> float:
    """``usage.cost`` when it is a finite number >= 0, else 0."""
    usage = body.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        return 0.0
    return float(cost)
