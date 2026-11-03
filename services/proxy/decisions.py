"""OpenAI Decisions API (``POST /v1/decisions``) on top of the proxy's backends.

Validates OpenAI's request shape, turns it into Laya ``/predict`` payloads (text, JSON and conversations) or
SigLIP2 ``/classify`` payloads (one image, choice questions only), and maps the backend answers back onto
OpenAI's ``Decision`` object. Reference: https://developers.openai.com/api/reference/resources/decisions/methods/create
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from typing import Any

from errors import ApiError
from models import MAX_IMAGE_B64_CHARS, MAX_IMAGE_BYTES, MAX_LABELS, MAX_QUESTIONS, MAX_STATE_BYTES

# OpenAI model names that mean "the default model": Laya's own script/language routing for text.
MODEL_ALIASES = frozenset({"gpt-6-luna"})
MAX_DECISION_QUESTIONS = 400
MAX_CHOICES = MAX_LABELS  # Laya criteria and SigLIP2 labels; OpenAI itself allows 255
MIN_LEVELS, MAX_LEVELS = 2, 10
MAX_INSTRUCTIONS = 2048
LAYA_CHUNK = MAX_QUESTIONS
MAX_LABEL_CHARS = 256

REQUEST_FIELDS = frozenset(
    {"model", "input", "questions", "prompt_cache_key", "prompt_cache_options", "safety_identifier"}
)
QUESTION_FIELDS = {
    "predicate": frozenset({"type", "name", "instructions"}),
    "choice": frozenset({"type", "name", "instructions", "choices"}),
    "score": frozenset({"type", "name", "instructions", "levels"}),
}
MESSAGE_PARTS = {
    "user": frozenset({"input_text", "input_image", "input_json"}),
    "developer": frozenset({"input_text"}),
    "system": frozenset({"input_text"}),
    "assistant": frozenset({"output_text", "refusal"}),
}
OUTPUT_PARTS = frozenset({"input_text", "input_image"})
CALLS = {"function_call": "function_call_output", "custom_tool_call": "custom_tool_call_output"}
LAYA_TYPES = {"predicate": "noul", "choice": "choice", "score": "score"}
QID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|jpg));base64,(.*)$", re.DOTALL)


def bad(param: str, message: str) -> ApiError:
    return ApiError("invalid_request", f"{param}: {message}", param=param)


@dataclass
class Image:
    base64: str
    media_type: str


@dataclass
class Question:
    type: str
    name: str | None
    instructions: str
    qid: str = ""
    # choice: (OpenAI value, backend label, description); score: (label, description)
    choices: list[tuple[str | bool, str, str | None]] = field(default_factory=list)
    levels: list[tuple[str, str | None]] = field(default_factory=list)


@dataclass
class Plan:
    model: str | None  # registry name to resolve; None means the task default
    questions: list[Question]
    state: str | dict | None = None  # Laya state (text path)
    image: Image | None = None  # SigLIP2 image (image path)


# ---------------------------------------------------------------- parsing


def _str(value: Any, param: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise bad(param, "must be a string")
    if not allow_empty and not value:
        raise bad(param, "must not be empty")
    return value


def _json_text(data: Any, param: str) -> str:
    if not (isinstance(data, dict) or (isinstance(data, list) and all(isinstance(d, dict) for d in data))):
        raise bad(param, "must be a JSON object or an array of objects")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _image(url: Any, param: str) -> Image:
    url = _str(url, param)
    if url.startswith(("http://", "https://")):
        raise bad(param, "hosted image URLs are not supported; send the image as a base64 data URL")
    m = DATA_URL.match(url)
    if m is None:
        raise bad(param, "must be a base64 data URL of a PNG or JPEG image (data:image/png;base64,...)")
    media_type, b64 = m.group(1), m.group(2)
    decoded = len(b64) * 3 // 4 - (2 if b64.endswith("==") else 1 if b64.endswith("=") else 0)
    if len(b64) > MAX_IMAGE_B64_CHARS or decoded > MAX_IMAGE_BYTES:
        raise ApiError(
            "payload_too_large", f"decoded image is {decoded} bytes; limit is {MAX_IMAGE_BYTES}", param=param
        )
    try:
        base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise bad(param, "not valid standard base64") from None
    return Image(b64, "image/jpeg" if media_type == "image/jpg" else media_type)


def _parts(content: Any, allowed: frozenset[str], param: str, images: list[tuple[Image, str]]) -> list[str]:
    """Render a content string or part list as text pieces; collect images."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise bad(param, "must be a string or an array of content parts")
    pieces = []
    for j, part in enumerate(content):
        p = f"{param}[{j}]"
        if not isinstance(part, dict):
            raise bad(p, "must be an object")
        kind = part.get("type")
        if kind not in allowed:
            raise bad(f"{p}.type", f"{kind!r} is not supported here (allowed: {', '.join(sorted(allowed))})")
        if kind in ("input_text", "output_text"):
            pieces.append(_str(part.get("text"), f"{p}.text"))
        elif kind == "refusal":
            pieces.append("refusal: " + _str(part.get("refusal"), f"{p}.refusal"))
        elif kind == "input_json":
            pieces.append(_json_text(part.get("data"), f"{p}.data"))
        else:  # input_image
            images.append((_image(part.get("image_url"), f"{p}.image_url"), f"{p}.image_url"))
    return pieces


def parse_input(value: Any) -> tuple[str | dict, list[tuple[Image, str]]]:
    """``input`` -> (Laya state, images). One user message stays plain text or a JSON object; more is a transcript."""
    images: list[tuple[Image, str]] = []
    if isinstance(value, str):
        return value, images
    if not isinstance(value, list):
        raise bad("input", "must be a string or an array of input items")
    if not value:
        raise bad("input", "must not be empty")

    lines: list[str] = []
    calls: dict[str, set[str]] = {kind: set() for kind in CALLS}
    for i, item in enumerate(value):
        p = f"input[{i}]"
        if not isinstance(item, dict):
            raise bad(p, "must be an object")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in MESSAGE_PARTS:
                raise bad(f"{p}.role", f"{role!r} is not supported (allowed: assistant, developer, system, user)")
            text = "\n".join(_parts(item.get("content"), MESSAGE_PARTS[role], f"{p}.content", images))
            lines.append(f"{'developer' if role == 'system' else role}: {text}")
        elif kind in CALLS:
            call_id = _str(item.get("call_id"), f"{p}.call_id", allow_empty=False)
            name = _str(item.get("name"), f"{p}.name", allow_empty=False)
            body = _str(item.get("arguments" if kind == "function_call" else "input"), p)
            calls[kind].add(call_id)
            lines.append(f"{kind} {call_id} {name}: {body}")
        elif kind in CALLS.values():
            call_id = _str(item.get("call_id"), f"{p}.call_id", allow_empty=False)
            call_kind = next(k for k, v in CALLS.items() if v == kind)
            if call_id not in calls[call_kind]:
                raise bad(f"{p}.call_id", f"no earlier {call_kind} has call_id {call_id!r}")
            text = "\n".join(_parts(item.get("output"), OUTPUT_PARTS, f"{p}.output", images))
            lines.append(f"{kind} {call_id}: {text}")
        else:
            raise bad(f"{p}.type", f"input item type {kind!r} is not supported")

    only = value[0] if len(value) == 1 else None
    if only is not None and only.get("type", "message") == "message" and only.get("role") == "user":
        content = only.get("content")
        if isinstance(content, str):
            return content, images
        kinds = [part["type"] for part in content]
        if kinds == ["input_json"] and isinstance(content[0]["data"], dict):
            return content[0]["data"], images
        if kinds and all(k == "input_text" for k in kinds):
            return "\n".join(part["text"] for part in content), images
    return "\n".join(lines), images


def _instructions(value: Any, param: str) -> str:
    if isinstance(value, list):
        if not value:
            raise bad(param, "must not be empty")
        if any(isinstance(part, dict) and part.get("type") == "input_image" for part in value):
            raise bad(param, "images in question instructions are not supported; put the image in input")
        images: list[tuple[Image, str]] = []
        value = "\n".join(_parts(value, frozenset({"input_text", "input_json"}), param, images))
    text = _str(value, param, allow_empty=False)
    if len(text) > MAX_INSTRUCTIONS:
        raise bad(param, f"at most {MAX_INSTRUCTIONS} characters are supported")
    return text


def _question(q: Any, i: int) -> Question:
    p = f"questions[{i}]"
    if not isinstance(q, dict):
        raise bad(p, "must be an object")
    kind = q.get("type")
    if kind not in QUESTION_FIELDS:
        raise bad(f"{p}.type", f"{kind!r} is not a question type (predicate, choice, score)")
    extra = sorted(set(q) - QUESTION_FIELDS[kind])
    if extra:
        raise bad(f"{p}.{extra[0]}", f"unknown field for a {kind} question")
    name = q.get("name")
    if name is not None:
        name = _str(name, f"{p}.name")
    question = Question(kind, name, _instructions(q.get("instructions"), f"{p}.instructions"))

    if kind == "choice":
        choices = q.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= MAX_CHOICES:
            raise bad(f"{p}.choices", f"must be an array of 2 to {MAX_CHOICES} choices")
        seen: set[str] = set()
        for j, c in enumerate(choices):
            cp = f"{p}.choices[{j}]"
            if not isinstance(c, dict) or set(c) - {"value", "description"}:
                raise bad(cp, "must be an object with value and an optional description")
            value = c.get("value")
            if not isinstance(value, (str, bool)):
                raise bad(f"{cp}.value", "must be a string or a boolean")
            label = ("true" if value else "false") if isinstance(value, bool) else value
            if label in seen:
                raise bad(
                    f"{p}.choices",
                    f"choice values must be unique ({label!r} repeats; a boolean and the "
                    "same text as a string count as the same value here)",
                )
            seen.add(label)
            desc = c.get("description")
            question.choices.append((value, label, None if desc is None else _str(desc, f"{cp}.description")))
    elif kind == "score":
        levels = q.get("levels")
        if not isinstance(levels, list) or not MIN_LEVELS <= len(levels) <= MAX_LEVELS:
            raise bad(f"{p}.levels", f"must be an array of {MIN_LEVELS} to {MAX_LEVELS} levels")
        for j, lv in enumerate(levels):
            lp = f"{p}.levels[{j}]"
            if not isinstance(lv, dict) or set(lv) - {"label", "description"}:
                raise bad(lp, "must be an object with label and an optional description")
            desc = lv.get("description")
            question.levels.append(
                (_str(lv.get("label"), f"{lp}.label"), None if desc is None else _str(desc, f"{lp}.description"))
            )
    return question


def _assign_qids(questions: list[Question]) -> None:
    """Laya question ids: the name when it is a valid, unused id; otherwise ``q<index>``."""
    used: set[str] = set()
    for i, q in enumerate(questions):
        qid = q.name if q.name is not None and QID.match(q.name) and q.name not in used else f"q{i}"
        while qid in used:
            qid += "_"
        used.add(qid)
        q.qid = qid


def parse(data: dict) -> Plan:
    extra = sorted(set(data) - REQUEST_FIELDS)
    if extra:
        raise bad(extra[0], "unknown parameter")
    model = data.get("model")
    if model is not None:
        model = _str(model, "model", allow_empty=False)
        if model in MODEL_ALIASES:
            model = None

    questions = data.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_DECISION_QUESTIONS:
        raise bad("questions", f"must be an array of 1 to {MAX_DECISION_QUESTIONS} questions")
    if "input" not in data:
        raise bad("input", "is required")
    state, images = parse_input(data["input"])
    plan = Plan(model, [_question(q, i) for i, q in enumerate(questions)])
    _assign_qids(plan.questions)

    if images:
        if len(images) > 1:
            raise bad("input", "at most one image is supported per request (it is classified with SigLIP2)")
        for i, q in enumerate(plan.questions):
            if q.type != "choice":
                raise bad(
                    f"questions[{i}]",
                    "with an image input only choice questions are supported "
                    "(SigLIP2 zero-shot classification over the choice values)",
                )
            for j, (_, label, _) in enumerate(q.choices):
                if not 1 <= len(label) <= MAX_LABEL_CHARS:
                    raise bad(
                        f"questions[{i}].choices[{j}].value",
                        f"image choice values must be 1 to {MAX_LABEL_CHARS} characters",
                    )
        plan.image = images[0][0]
        return plan

    if isinstance(state, str):
        if not state.strip():
            raise bad("input", "must contain text or JSON evidence")
        size = len(state.encode("utf-8"))
    else:
        size = len(_json_text(state, "input").encode("utf-8"))
    if size > MAX_STATE_BYTES:
        raise ApiError("payload_too_large", f"input is {size} bytes as text; limit is {MAX_STATE_BYTES}", param="input")
    plan.state = state
    return plan


# ---------------------------------------------------------------- backend payloads


def _laya_question(q: Question) -> dict:
    out: dict[str, Any] = {"type": LAYA_TYPES[q.type], "instructions": q.instructions}
    if q.type == "choice":
        out["criteria"] = {label: label if desc is None else desc for _, label, desc in q.choices}
    elif q.type == "score":
        out["criteria"] = [label if desc is None else f"{label}: {desc}" for label, desc in q.levels]
    return out


def laya_payloads(plan: Plan) -> list[dict]:
    """Laya ``/predict`` bodies, at most ``LAYA_CHUNK`` questions each, in question order."""
    qs = plan.questions
    return [
        {"state": plan.state, "questions": {q.qid: _laya_question(q) for q in qs[i : i + LAYA_CHUNK]}}
        for i in range(0, len(qs), LAYA_CHUNK)
    ]


def siglip_payloads(plan: Plan) -> list[dict]:
    """One SigLIP2 ``/classify`` body per (choice) question."""
    assert plan.image is not None
    image = {"base64": plan.image.base64, "media_type": plan.image.media_type}
    return [{"image": image, "candidate_labels": [label for _, label, _ in q.choices]} for q in plan.questions]


# ---------------------------------------------------------------- answers


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError("backend_error", "backend returned an unexpected answer")
    return value


def _from_laya(q: Question, ans: Any) -> dict:
    if not isinstance(ans, dict):
        return {"type": "refusal", "name": q.name}
    if q.type == "predicate":
        return {"type": "predicate", "name": q.name, "probability": _number(ans.get("noul"))}
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        raise ApiError("backend_error", "backend returned an unexpected answer")
    confidence = _number(ans.get("confidence"))
    if q.type == "choice":
        by_label = {label: value for value, label, _ in q.choices}
        if ans.get("choice") not in by_label:
            raise ApiError("backend_error", "backend returned a choice that was not offered")
        return {
            "type": "choice",
            "name": q.name,
            "choice": by_label[ans["choice"]],
            "probabilities": [
                {"value": value, "probability": _number(probs.get(label, 0.0))} for value, label, _ in q.choices
            ],
            "confidence": confidence,
        }
    return {
        "type": "score",
        "name": q.name,
        "score": _number(ans.get("score")),
        "probabilities": [
            {"value": i, "label": label, "probability": _number(probs.get(str(i), 0.0))}
            for i, (label, _) in enumerate(q.levels)
        ],
        "confidence": confidence,
    }


def _from_siglip(q: Question, results: Any) -> dict:
    if not isinstance(results, list) or not results:
        raise ApiError("backend_error", "image backend returned no labels")
    scores = {r["label"]: _number(r["score"]) for r in results}
    by_label = {label: value for value, label, _ in q.choices}
    top = max(scores, key=scores.__getitem__)
    if top not in by_label:
        raise ApiError("backend_error", "image backend returned a label that was not offered")
    return {
        "type": "choice",
        "name": q.name,
        "choice": by_label[top],
        "probabilities": [{"value": value, "probability": scores.get(label, 0.0)} for value, label, _ in q.choices],
        "confidence": scores[top],
    }


def usage(input_tokens: int = 0, output_tokens: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def render_text(plan: Plan, model: str, chunks: list[tuple[dict, dict]]) -> dict:
    """``chunks``: (Laya answers by qid, meta) per Laya call."""
    answers: dict[str, Any] = {}
    tokens_in = tokens_out = 0
    for results, meta in chunks:
        answers.update(results)
        u = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
        tokens_in += u.get("input_tokens", 0) if isinstance(u.get("input_tokens"), int) else 0
        tokens_out += u.get("output_tokens", 0) if isinstance(u.get("output_tokens"), int) else 0
    return {
        "model": model,
        "answers": [_from_laya(q, answers.get(q.qid)) for q in plan.questions],
        "usage": usage(tokens_in, tokens_out),
    }


def render_image(plan: Plan, model: str, results: list[Any]) -> dict:
    return {
        "model": model,
        "answers": [_from_siglip(q, r) for q, r in zip(plan.questions, results, strict=True)],
        "usage": usage(),
    }
