"""POST /api/alpha/decisions: the OpenRouter Decisions wire format, served by Laya, SigLIP2 or OpenRouter."""

from __future__ import annotations

import base64
import json
import logging
import re

import httpx
import pytest
from fastapi.testclient import TestClient
from test_decisions import laya_echo

from app import create_app
from config import Settings
from conftest import API_KEY, IMAGE_URL, TEXT_URL
from orkit import JEV_RESPONSE, OPENROUTER_KEY, FakeOpenRouter, write_key, write_registry

PATH = "/api/alpha/decisions"
PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
PNG_URL = f"data:image/png;base64,{PNG_B64}"
WEBP_URL = "data:image/webp;base64," + base64.b64encode(b"RIFF0000WEBPVP8 ").decode()
AUTH = {"Authorization": f"Bearer {API_KEY}"}

TICKET = {"customer_tier": "enterprise", "ticket": "My checkout page shows a blank screen after I click Pay."}
QUESTIONS = {
    "is_bug": {
        "type": "noul",
        "instructions": "Is the customer reporting a software defect?",
        "criteria": {
            "true": "The customer describes broken or unexpected product behavior.",
            "false": "The customer is asking a question or requesting a feature.",
        },
    },
    "team": {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {"account": "Login, permissions, or profile issues.", "payments": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Can wait", {"level": "this week"}, "Blocking revenue right now"],
    },
}


@pytest.fixture
def fake_or():
    return FakeOpenRouter()


@pytest.fixture
def make(tmp_path, backends, fake_or):
    backends.text = laya_echo

    def _make(**overrides) -> TestClient:
        base = {
            "api_keys": (API_KEY,),
            "text_backend_url": TEXT_URL,
            "image_backend_url": IMAGE_URL,
            "registry_file": write_registry(tmp_path),
            "openrouter_key_file": write_key(tmp_path),
            "file_keys_allow_paid": True,
        }
        base.update(overrides)
        app = create_app(Settings(**base), transport=httpx.MockTransport(fake_or.wrap(backends.handler)))
        c = TestClient(app, raise_server_exceptions=False)
        c.__enter__()
        clients.append(c)
        return c

    clients: list[TestClient] = []
    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def c(make):
    return make()


def post(client, body, headers=AUTH):
    return client.post(PATH, json=body, headers=headers)


def predict_calls(backends) -> list[dict]:
    return [json.loads(r.content) for r in backends.requests if r.url.path == "/predict"]


def classify_calls(backends) -> list[dict]:
    return [json.loads(r.content) for r in backends.requests if r.url.path == "/classify"]


def or_error(resp, status: int, code: str, param="__any__") -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    err = body["error"]
    assert err["code"] == status
    assert isinstance(err["message"], str) and err["message"]
    assert err["metadata"]["code"] == code
    if param != "__any__":
        assert err["metadata"]["param"] == param
    return err


# ---------------------------------------------------------------- request validation


@pytest.mark.parametrize(
    "body,param",
    [
        ({"state": "x", "questions": {"q": {"type": "noul"}}}, "model"),
        ({"model": 3, "state": "x", "questions": {"q": {"type": "noul"}}}, "model"),
        ({"model": "laya-english", "questions": {"q": {"type": "noul"}}}, "state"),
        ({"model": "laya-english", "state": 5, "questions": {"q": {"type": "noul"}}}, "state"),
        ({"model": "laya-english", "state": "x"}, "questions"),
        ({"model": "laya-english", "state": "x", "questions": {}}, "questions"),
        ({"model": "laya-english", "state": "x", "questions": {"bad id!": {"type": "noul"}}}, "questions"),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "predicate"}}}, "questions.q.type"),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul", "x": 1}}}, "questions.q.x"),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul", "criteria": {"maybe": "?"}}}},
            "questions.q.criteria.maybe",
        ),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul", "instructions": 5}}},
            "questions.q.instructions",
        ),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "choice"}}}, "questions.q.criteria"),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
            "questions.q.criteria",
        ),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "score", "criteria": []}}},
            "questions.q.criteria",
        ),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "score", "criteria": ["a", None]}}},
            "questions.q.criteria[1]",
        ),
        ({"model": "jev", "state": "x", "questions": {"q": {"type": "box"}}}, "questions.q.instructions"),
        (
            {"model": "jev", "state": "x", "questions": {"q": {"type": "box", "instructions": "a", "criteria": {}}}},
            "questions.q.criteria",
        ),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}, "stream": True}, "stream"),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}, "user": "u" * 257}, "user"),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}, "session_id": 7},
            "session_id",
        ),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}, "trace": "t"}, "trace"),
        ({"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}, "provider": 1}, "provider"),
    ],
)
def test_invalid_requests(c, backends, body, param):
    or_error(post(c, body), 400, "invalid_request", param)
    assert predict_calls(backends) == []


def test_body_must_be_a_json_object(c):
    or_error(c.post(PATH, content=b"[1]", headers=AUTH), 400, "invalid_request")
    or_error(c.post(PATH, content=b'{"model": NaN}', headers=AUTH), 400, "invalid_request")


def test_unused_asset_names_the_key(c, fake_or):
    body = {
        "model": "jev",
        "state": ["receipt.png"],
        "questions": {"q": {"type": "noul", "instructions": "Is it paid?"}},
        "assets": {
            "receipt.png": {"format": "png", "data_base64": PNG_B64},
            "extra.png": {"type": "image", "format": "png", "data_base64": PNG_B64},
        },
    }
    err = or_error(post(c, body), 400, "invalid_request", "assets.extra.png")
    assert "extra.png" in err["message"]
    assert fake_or.requests == []


@pytest.mark.parametrize(
    "asset,param",
    [
        ({"format": "gif", "data_base64": PNG_B64}, "assets.a.format"),
        ({"format": "png", "data_base64": "not base64!"}, "assets.a.data_base64"),
        ({"format": "png", "data_base64": ""}, "assets.a.data_base64"),
        ({"format": "png", "data_base64": PNG_B64, "part": 1}, "assets.a.part"),
        ({"type": "pdf", "format": "png", "data_base64": PNG_B64}, "assets.a.type"),
    ],
)
def test_invalid_assets(c, asset, param):
    body = {"model": "jev", "state": "a", "questions": {"q": {"type": "noul"}}, "assets": {"a": asset}}
    or_error(post(c, body), 400, "invalid_request", param)


def test_asset_referenced_inside_text_counts_as_used(c, fake_or):
    body = {
        "model": "jev",
        "state": {"note": "see the photo in shot-1 for details"},
        "questions": {"q": {"type": "box", "instructions": "the price tag"}},
        "assets": {"shot-1": {"format": "jpg", "data_base64": PNG_B64}},
    }
    assert post(c, body).status_code == 200
    assert fake_or.bodies()[0]["assets"] == body["assets"]


def test_hosted_image_urls_are_rejected(c, fake_or):
    body = {
        "model": "jev",
        "state": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        "questions": {"q": {"type": "noul"}},
    }
    err = or_error(post(c, body), 400, "invalid_request", "state[0].image_url.url")
    assert "data URL" in err["message"]
    assert fake_or.requests == []


def test_malformed_image_part_is_rejected(c):
    for part in (
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}},
        {"type": "image_url", "image_url": {"url": "data:image/gif;base64," + PNG_B64}},
        {"type": "image_url"},
    ):
        or_error(
            post(c, {"model": "jev", "state": [part], "questions": {"q": {"type": "noul"}}}), 400, "invalid_request"
        )


# ---------------------------------------------------------------- model resolution and capabilities


def test_unknown_model_lists_models_and_aliases(c):
    err = or_error(
        post(c, {"model": "gpt-9", "state": "x", "questions": {"q": {"type": "noul"}}}), 400, "invalid_request", "model"
    )
    for name in ("laya-english", "jev", "auto", "vision"):
        assert name in err["message"]


def test_auto_routes_text_to_laya(c, backends, fake_or):
    resp = post(c, {"model": "auto", "state": "hello", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "laya-english"
    assert len(predict_calls(backends)) == 1 and fake_or.requests == []


def test_auto_routes_images_to_the_image_target(c, backends, fake_or):
    body = {
        "model": "auto",
        "state": [{"type": "image_url", "image_url": {"url": PNG_URL}}],
        "questions": {"q": {"type": "noul", "instructions": "Is there a dog?"}},
    }
    resp = post(c, body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "jev"
    assert fake_or.bodies()[0]["model"] == "typesafe/jev-1.14"
    assert predict_calls(backends) == []


def test_auto_routes_box_questions_and_assets_to_the_image_target(c, fake_or):
    box = {"model": "auto", "state": "a shelf", "questions": {"q": {"type": "box", "instructions": "the red can"}}}
    assert post(c, box).json()["model"] == "jev"
    asset = {
        "model": "auto",
        "state": "pic",
        "questions": {"q": {"type": "noul"}},
        "assets": {"pic": {"format": "png", "data_base64": PNG_B64}},
    }
    assert post(c, asset).json()["model"] == "jev"
    assert len(fake_or.requests) == 2


def test_laya_checkpoint_names_are_accepted(c, backends):
    resp = post(c, {"model": "english", "state": "x", "questions": {"q": {"type": "noul"}}})
    assert resp.status_code == 200 and resp.json()["model"] == "laya-english"
    assert predict_calls(backends)[0]["model"] == "english"


@pytest.mark.parametrize(
    "body,needle",
    [
        (
            {
                "model": "laya-english",
                "state": [{"type": "image_url", "image_url": {"url": PNG_URL}}],
                "questions": {"q": {"type": "noul"}},
            },
            "model 'laya-english' does not accept images",
        ),
        (
            {"model": "laya-english", "state": "x", "questions": {"q": {"type": "box", "instructions": "a can"}}},
            "model 'laya-english' does not accept box questions",
        ),
        (
            {
                "model": "jev",
                "state": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}],
                "questions": {"q": {"type": "noul"}},
            },
            "model 'jev' does not accept audio",
        ),
        (
            {
                "model": "jev",
                "state": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}],
                "questions": {"q": {"type": "noul"}},
            },
            "model 'jev' does not accept video",
        ),
        (
            {
                "model": "vision",
                "state": [{"type": "image_url", "image_url": {"url": PNG_URL}}],
                "questions": {"q": {"type": "noul"}},
            },
            "choice questions",
        ),
        (
            {
                "model": "vision",
                "state": [{"type": "image_url", "image_url": {"url": PNG_URL}}, "and some text"],
                "questions": {"q": {"type": "choice", "criteria": {"dog": None, "cat": None}}},
            },
            "exactly one image",
        ),
        (
            {
                "model": "vision",
                "state": "just text",
                "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
            },
            "exactly one image",
        ),
        (
            {
                "model": "vision",
                "state": [{"type": "image_url", "image_url": {"url": WEBP_URL}}],
                "questions": {"q": {"type": "choice", "criteria": {"dog": None, "cat": None}}},
            },
            "PNG or JPEG",
        ),
    ],
)
def test_capability_errors(c, backends, fake_or, body, needle):
    err = or_error(post(c, body), 400, "invalid_request")
    assert needle in err["message"]
    assert backends.requests == [] and fake_or.requests == []


# ---------------------------------------------------------------- paid guard


def test_paid_model_needs_paid_access(make, fake_or):
    c = make(file_keys_allow_paid=False)
    body = {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}}
    err = or_error(post(c, body), 403, "forbidden")
    assert "not allowed to use paid model 'jev'" in err["message"]
    err = or_error(
        post(c, dict(body, model="auto", questions={"q": {"type": "box", "instructions": "a"}})), 403, "forbidden"
    )
    assert fake_or.requests == []
    assert post(c, dict(body, model="laya-english")).status_code == 200  # free models stay available


def test_paid_access_with_sqlite_keys_applies_immediately(tmp_path, backends, fake_or, clock):
    import keyservice as ks
    from adminkit import make_admin

    backends.text = laya_echo

    class Routed:  # make_admin builds its transport from .handler
        handler = staticmethod(fake_or.wrap(backends.handler))

    a = make_admin(
        tmp_path, Routed, clock, registry_file=write_registry(tmp_path), openrouter_key_file=write_key(tmp_path)
    )
    try:
        owner = ks.get_viewer(a.db, "owner@example.com")
        key = ks.create_key(a.db, owner, "k", now=clock())
        hdr = {"Authorization": f"Bearer {key['key']}"}
        body = {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}}
        with TestClient(a.app, raise_server_exceptions=False) as c:
            or_error(post(c, body, hdr), 403, "forbidden")
            ks.set_paid_access(a.db, owner, key["id"], True, now=clock())
            resp = post(c, body, hdr)
            assert resp.status_code == 200, resp.text
            ks.set_paid_access(a.db, owner, key["id"], False, now=clock())
            or_error(post(c, body, hdr), 403, "forbidden")
            assert c.app.state.usage.flush()
        row = a.db.conn.execute(
            "SELECT requests, client_errors, cost_usd_sum FROM usage_daily WHERE model = 'jev'"
        ).fetchone()
        assert (row[0], row[1]) == (3, 2) and row[2] == pytest.approx(JEV_RESPONSE["usage"]["cost"])
    finally:
        a.close()


def test_missing_openrouter_key_is_503(make, fake_or):
    c = make(openrouter_key_file=None)
    or_error(post(c, {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}}), 503, "backend_unavailable")
    assert fake_or.requests == []


# ---------------------------------------------------------------- openrouter adapter


def test_forwards_to_openrouter(c, fake_or, caplog):
    caplog.set_level(logging.DEBUG)
    body = {
        "model": "jev",
        "state": [{"type": "image_url", "image_url": {"url": PNG_URL}}, "receipt"],
        "questions": dict(QUESTIONS, where={"type": "box", "instructions": {"target": "the total"}}),
        "assets": {"receipt": {"format": "png", "data_base64": PNG_B64}},
        "provider": {"allow_fallbacks": False},
        "session_id": "s-1",
        "trace": {"trace_id": "t-1"},
        "user": "u-1",
    }
    resp = post(c, body)
    assert resp.status_code == 200, resp.text
    (req,) = fake_or.requests
    assert str(req.url) == "https://openrouter.ai/api/alpha/decisions"
    assert req.headers["authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert req.headers["http-referer"] == "https://decide.ai.huynle.com"
    assert req.headers["x-title"] == "Homelab Decision API"
    sent = json.loads(req.content)
    assert sent == {
        "model": "typesafe/jev-1.14",
        "state": body["state"],
        "questions": body["questions"],
        "assets": body["assets"],
        "provider": body["provider"],
    }
    out = resp.json()
    assert out == {
        "model": "jev",
        "answers": JEV_RESPONSE["answers"],
        "usage": JEV_RESPONSE["usage"],
        "id": JEV_RESPONSE["id"],
        "provider": "TypeSafe",
    }
    assert OPENROUTER_KEY not in resp.text
    assert all(OPENROUTER_KEY not in r.getMessage() for r in caplog.records)
    assert all(API_KEY not in r.headers.get("authorization", "") for r in fake_or.requests)


def test_referer_is_configurable(make, fake_or):
    c = make(openrouter_referer="https://example.test")
    post(c, {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}})
    assert fake_or.requests[0].headers["http-referer"] == "https://example.test"


def test_annotations_pass_through(c, fake_or):
    ann = [{"type": "url_citation", "url_citation": {"url": "https://example.com"}}]
    fake_or.reply = lambda req: httpx.Response(200, json=dict(JEV_RESPONSE, annotations=ann))
    assert post(c, {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}}).json()["annotations"] == ann


def _raise(exc):
    def reply(req):
        raise exc

    return reply


@pytest.mark.parametrize(
    "reply,status,code",
    [
        (
            lambda r: httpx.Response(400, json={"error": {"code": 400, "message": "bad box /srv/x/y"}}),
            400,
            "invalid_request",
        ),
        (lambda r: httpx.Response(422, json={"error": {"message": "nope"}}), 400, "invalid_request"),
        (lambda r: httpx.Response(413, json={"error": {"message": "too big"}}), 413, "payload_too_large"),
        (lambda r: httpx.Response(401, json={"error": {"message": "No auth credentials found"}}), 502, "backend_error"),
        (lambda r: httpx.Response(402, json={"error": {"message": "Insufficient credits"}}), 502, "backend_error"),
        (lambda r: httpx.Response(403, json={"error": {"message": "key disabled"}}), 502, "backend_error"),
        (lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}), 429, "rate_limited"),
        (lambda r: httpx.Response(500), 502, "backend_error"),
        (lambda r: httpx.Response(502), 502, "backend_error"),
        (lambda r: httpx.Response(524), 502, "backend_error"),
        (lambda r: httpx.Response(529), 502, "backend_error"),
        (lambda r: httpx.Response(503), 503, "backend_unavailable"),
        (_raise(httpx.ReadTimeout("slow")), 504, "backend_timeout"),
        (_raise(httpx.ConnectError("down")), 503, "backend_unavailable"),
        (lambda r: httpx.Response(200, text="<html>"), 502, "backend_error"),
        (lambda r: httpx.Response(200, text='{"answers": {"q": {"type": "noul", "noul": NaN}}}'), 502, "backend_error"),
        (lambda r: httpx.Response(200, json={"answers": ["x"]}), 502, "backend_error"),
        (lambda r: httpx.Response(200, json={"answers": {"q": 1}}), 502, "backend_error"),
    ],
)
def test_upstream_error_mapping(c, fake_or, reply, status, code):
    fake_or.reply = reply
    err = or_error(post(c, {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}}), status, code)
    if status == 400:
        assert "[path]" in err["message"] or "nope" in err["message"]
    if code == "backend_error":
        assert "OpenRouter" in err["message"] or "upstream" in err["message"]
    assert "No auth credentials" not in err["message"] and "Insufficient" not in err["message"]


# ---------------------------------------------------------------- local translation: Laya


def test_laya_round_trip(c, backends):
    resp = post(c, {"model": "laya-english", "state": TICKET, "questions": QUESTIONS, "session_id": "s"})
    assert resp.status_code == 200, resp.text
    (sent,) = predict_calls(backends)
    assert sent["state"] == TICKET
    assert sent["model"] == "english"
    is_bug = sent["questions"]["is_bug"]
    assert is_bug["type"] == "noul" and "criteria" not in is_bug
    assert is_bug["instructions"].startswith("Is the customer reporting a software defect?")
    assert "True if: The customer describes broken" in is_bug["instructions"]
    assert "False if: The customer is asking" in is_bug["instructions"]
    assert sent["questions"]["team"] == {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {"account": "Login, permissions, or profile issues.", "payments": "payments"},
    }
    assert sent["questions"]["urgency"]["criteria"] == [
        "Can wait",
        '{"level":"this week"}',
        "Blocking revenue right now",
    ]

    out = resp.json()
    assert out["model"] == "laya-english"
    assert out["provider"] == "homelab"
    assert re.fullmatch(r"gen-dec-\d{10}-[A-Za-z0-9]{20}", out["id"])
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 0, "cost": 0}
    assert out["answers"]["is_bug"] == {"type": "noul", "noul": 0.8}
    team = out["answers"]["team"]
    assert team["type"] == "choice" and team["choice"] == "account" and team["confidence"] == 0.6
    assert set(team["probabilities"]) == {"account", "payments"}
    urgency = out["answers"]["urgency"]
    assert urgency["type"] == "score" and urgency["score"] == 2.0 and urgency["confidence"] == 0.9
    assert urgency["probabilities"] == {"0": 0.0, "1": 0.0, "2": 1.0}
    assert urgency["legend"] == {"0": "Can wait", "1": {"level": "this week"}, "2": "Blocking revenue right now"}


def test_laya_state_arrays(c, backends):
    post(c, {"model": "laya-english", "state": ["line one", "line two"], "questions": {"q": {"type": "noul"}}})
    post(c, {"model": "laya-english", "state": ["a", {"b": 1}], "questions": {"q": {"type": "noul"}}})
    first, second = predict_calls(backends)
    assert first["state"] == "line one\nline two"
    assert second["state"] == '["a",{"b":1}]'


def test_laya_object_instructions_and_fallback(c, backends):
    post(
        c,
        {
            "model": "laya-english",
            "state": "x",
            "questions": {
                "a": {"type": "noul", "instructions": {"ask": "is it red?"}},
                "b": {"type": "noul", "instructions": None},
                "c": {"type": "choice", "criteria": {"x": {"desc": "ex"}, "y": ["why"]}},
            },
        },
    )
    (sent,) = predict_calls(backends)
    assert sent["questions"]["a"]["instructions"] == '{"ask":"is it red?"}'
    assert sent["questions"]["b"]["instructions"] == "Answer this question about the state."
    assert sent["questions"]["c"]["instructions"] == "Answer this question about the state."
    assert sent["questions"]["c"]["criteria"] == {"x": '{"desc":"ex"}', "y": '["why"]'}


def test_more_than_32_questions_are_split(c, backends):
    questions = {f"q{i:02d}": {"type": "noul", "instructions": f"question {i}"} for i in range(70)}
    resp = post(c, {"model": "laya-english", "state": "x", "questions": questions})
    assert resp.status_code == 200, resp.text
    calls = predict_calls(backends)
    assert [len(call["questions"]) for call in calls] == [32, 32, 6]
    assert list(resp.json()["answers"]) == list(questions)
    assert resp.json()["usage"]["input_tokens"] == 30


@pytest.mark.parametrize(
    "question,needle",
    [
        ({"type": "choice", "criteria": {"only": None}}, "2 to 64"),
        ({"type": "choice", "criteria": {f"o{i}": None for i in range(65)}}, "2 to 64"),
        ({"type": "score", "criteria": ["one"]}, "2 to 16"),
        ({"type": "score", "criteria": [str(i) for i in range(17)]}, "2 to 16"),
        ({"type": "noul", "instructions": "x" * 3000}, "2048"),
    ],
)
def test_local_limits(c, backends, question, needle):
    err = or_error(
        post(c, {"model": "laya-english", "state": "x", "questions": {"q": question}}), 400, "invalid_request"
    )
    assert needle in err["message"]
    assert predict_calls(backends) == []


def test_local_state_limit(c):
    or_error(
        post(c, {"model": "laya-english", "state": "x" * 40000, "questions": {"q": {"type": "noul"}}}),
        413,
        "payload_too_large",
    )
    or_error(
        post(c, {"model": "laya-english", "state": "", "questions": {"q": {"type": "noul"}}}), 400, "invalid_request"
    )


def test_missing_laya_answer_is_a_refusal(c, backends):
    backends.text = lambda req: httpx.Response(200, json={"answers": {}, "usage": {"input_tokens": 3}})
    resp = post(c, {"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}})
    assert resp.json()["answers"] == {"q": {"type": "refusal"}}


def test_non_finite_or_unoffered_laya_answers_are_502(c, backends):
    backends.text = lambda req: httpx.Response(
        200, json={"answers": {"q": {"type": "choice", "choice": "zzz", "probabilities": {}, "confidence": 1}}}
    )
    body = {
        "model": "laya-english",
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
    }
    or_error(post(c, body), 502, "backend_error")


# ---------------------------------------------------------------- local translation: SigLIP2


@pytest.mark.parametrize(
    "state,assets",
    [
        ([{"type": "image_url", "image_url": {"url": PNG_URL}}], None),
        ({"type": "image_url", "image_url": {"url": PNG_URL}}, None),
        ("photo", {"photo": {"format": "png", "data_base64": PNG_B64}}),
        (["photo"], {"photo": {"format": "jpeg", "data_base64": PNG_B64}}),
    ],
)
def test_siglip_path(c, backends, state, assets):
    def siglip_echo(request):
        labels = json.loads(request.content)["candidate_labels"]
        scores = [0.7] + [round(0.3 / (len(labels) - 1), 4)] * (len(labels) - 1)
        results = [{"label": label, "score": sc} for label, sc in zip(labels, scores, strict=True)]
        return httpx.Response(200, json={"model": "siglip2", "results": results})

    backends.image = siglip_echo
    body = {
        "model": "vision",
        "state": state,
        "questions": {
            "breed": {"type": "choice", "criteria": {"german shepherd": None, "dog": "any dog", "cat": None}},
            "kind": {"type": "choice", "criteria": {"dog": None, "cat": None}},
        },
    }
    if assets:
        body["assets"] = assets
    resp = post(c, body)
    assert resp.status_code == 200, resp.text
    calls = classify_calls(backends)
    assert [call["candidate_labels"] for call in calls] == [["german shepherd", "dog", "cat"], ["dog", "cat"]]
    assert calls[0]["image"]["base64"] == PNG_B64
    out = resp.json()
    assert out["model"] == "siglip2-base-patch16-224" and out["provider"] == "homelab"
    assert out["usage"] == {"input_tokens": 0, "output_tokens": 0, "cost": 0}
    breed = out["answers"]["breed"]
    assert breed["type"] == "choice" and breed["choice"] == "german shepherd"
    assert set(breed["probabilities"]) == {"german shepherd", "dog", "cat"}
    assert breed["confidence"] == breed["probabilities"]["german shepherd"]


# ---------------------------------------------------------------- envelope, limits, metrics, scope


def test_unauthorized_uses_openrouter_envelope(c):
    resp = c.post(PATH, json={"model": "jev"})
    err = or_error(resp, 401, "unauthorized", None)
    assert resp.headers["www-authenticate"].startswith("Bearer")
    assert err["message"] == "missing or invalid API key"


def test_v1_errors_keep_their_envelope(c):
    body = c.get("/v1/models").json()
    assert body["error"]["code"] == "unauthorized" and "latency_ms" in body


def test_body_limit_is_16_mib(c):
    big = b'{"model": "jev", "state": "' + b"x" * (16 * 1024 * 1024) + b'"}'
    or_error(c.post(PATH, content=big, headers=AUTH), 413, "payload_too_large")


def test_large_image_fits_under_the_body_limit(c, fake_or):
    data = base64.b64encode(b"\x89PNG" + b"\x00" * (10 * 1024 * 1024)).decode()
    body = {"model": "jev", "state": f"data:image/png;base64,{data}", "questions": {"q": {"type": "noul"}}}
    body["state"] = [{"type": "image_url", "image_url": {"url": body["state"]}}]
    assert post(c, body).status_code == 200


def test_metrics_and_cost_counter(c, fake_or):
    from metrics import generate_latest

    post(c, {"model": "jev", "state": "x", "questions": {"q": {"type": "noul"}}})
    post(c, {"model": "laya-english", "state": "x", "questions": {"q": {"type": "noul"}}})
    text = generate_latest(c.app.state.metrics.registry).decode()
    assert 'decision_api_requests_total{code="200",key_id="' in text
    assert 'route="/api/alpha/decisions"' in text
    assert re.search(r'decision_api_upstream_cost_usd_total\{model="jev"\} 1\.9992e-05', text), text
    assert 'decision_api_backend_request_duration_seconds_count{model="jev"} 1.0' in text


def test_v1_routes_never_reach_openrouter(c, fake_or):
    for model in ("jev", "auto"):
        r1 = c.post(
            "/v1/decide/text",
            json={"model": model, "state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}},
            headers=AUTH,
        )
        assert r1.status_code == 400 and r1.json()["error"]["code"] == "invalid_request"
        r2 = c.post(
            "/v1/decisions",
            json={"model": model, "input": "x", "questions": [{"type": "predicate", "instructions": "?"}]},
            headers=AUTH,
        )
        assert r2.status_code == 400
    assert fake_or.requests == []
