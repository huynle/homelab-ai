"""Behaviour of the decision proxy against mocked backends (no network)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from conftest import API_KEY, IMAGE_URL, OTHER_KEY, TEXT_URL

TEXT = "/v1/decide/text"
IMAGE = "/v1/classify/image"


def assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    return body


# ---------------------------------------------------------------- text


def test_text_success_maps_laya_into_envelope(client, auth, laya_request, laya_predict, backends):
    resp = client.post(TEXT, json=laya_request, headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "laya-english"
    assert body["model_version"] == "laya-serve-0.4.20"
    assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] >= 0
    assert body["results"] == laya_predict["answers"]
    assert body["meta"]["backend_model"] == "laya-rl-agent"
    assert body["meta"]["usage"] == laya_predict["usage"]
    expected_routing = {k: v for k, v in laya_predict["routing"].items() if k != "repo"}
    assert body["meta"]["routing"] == expected_routing
    assert "repo" not in body["meta"]["routing"]
    assert "/home/" not in resp.text


def test_text_forwards_state_questions_and_model_only(client, auth, laya_request, backends):
    payload = dict(laya_request, model="english", min_confidence=0.5)
    client.post(TEXT, json=payload, headers=auth)
    (sent,) = [r for r in backends.requests if r.url.path == "/predict"]
    assert str(sent.url) == f"{TEXT_URL}/predict"
    assert sent.method == "POST"
    forwarded = json.loads(sent.content)
    assert forwarded == {
        "state": laya_request["state"],
        "questions": laya_request["questions"],
        "model": "english",
    }


def test_text_without_model_does_not_send_model(client, auth, laya_request, backends):
    client.post(TEXT, json=laya_request, headers=auth)
    forwarded = json.loads(backends.requests[-1].content)
    assert "model" not in forwarded


def test_client_authorization_is_not_forwarded(client, auth, laya_request, image_request, backends):
    client.post(TEXT, json=laya_request, headers=auth)
    client.post(IMAGE, json=image_request, headers=auth)
    assert len(backends.requests) == 2
    for req in backends.requests:
        assert "authorization" not in {k.lower() for k in req.headers.keys()}
        assert API_KEY not in req.content.decode()


def test_min_confidence_flags_low_answers(client, auth, laya_request):
    # fixture confidences: department 0.9141, urgency 0.1492, churn_risk 0.8184
    resp = client.post(TEXT, json=dict(laya_request, min_confidence=0.85), headers=auth)
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert results["department"]["low_confidence"] is False
    assert results["urgency"]["low_confidence"] is True
    assert results["churn_risk"]["low_confidence"] is True


def test_no_min_confidence_means_no_flag(client, auth, laya_request):
    results = client.post(TEXT, json=laya_request, headers=auth).json()["results"]
    assert all("low_confidence" not in a for a in results.values())


def test_text_string_state_is_accepted(client, auth, laya_request, backends):
    payload = dict(laya_request, state="please refund me")
    assert client.post(TEXT, json=payload, headers=auth).status_code == 200
    assert json.loads(backends.requests[-1].content)["state"] == "please refund me"


def test_text_more_than_32_questions_is_400(client, auth, laya_request, backends):
    q = {"type": "noul", "instructions": "yes?"}
    payload = dict(laya_request, questions={f"q{i}": q for i in range(33)})
    body = assert_error(client.post(TEXT, json=payload, headers=auth), 400, "invalid_request")
    assert "32" in body["error"]["message"]
    assert backends.requests == []


def test_text_exactly_32_questions_is_ok(client, auth, laya_request):
    q = {"type": "noul", "instructions": "yes?"}
    payload = dict(laya_request, questions={f"q{i}": q for i in range(32)})
    assert client.post(TEXT, json=payload, headers=auth).status_code == 200


def test_text_state_over_32_kib_is_413(client, auth, laya_request, backends):
    payload = dict(laya_request, state={"body": "x" * 33000})
    assert_error(client.post(TEXT, json=payload, headers=auth), 413, "payload_too_large")
    assert backends.requests == []


def test_text_body_over_64_kib_is_413(client, auth, laya_request, backends):
    payload = dict(laya_request, padding="x" * (64 * 1024))
    assert_error(client.post(TEXT, json=payload, headers=auth), 413, "payload_too_large")
    assert backends.requests == []


def test_text_body_limit_enforced_without_content_length(client, auth, backends):
    def chunks():
        for _ in range(80):
            yield b"x" * 1024

    resp = client.post(TEXT, content=chunks(), headers={**auth, "Content-Type": "application/json"})
    assert_error(resp, 413, "payload_too_large")


@pytest.mark.parametrize(
    "payload",
    [
        {"questions": {"a": {"type": "noul", "instructions": "x"}}},
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"a": {"type": "maybe", "instructions": "x"}}},
        {"state": "x", "questions": {"a": {"type": "choice", "instructions": "x"}}},
        {"state": "x", "questions": {"a": {"type": "noul", "instructions": "x"}}, "extra": 1},
        {"state": "x", "questions": {"a": {"type": "noul", "instructions": "x"}}, "min_confidence": 2},
        {"state": "", "questions": {"a": {"type": "noul", "instructions": "x"}}},
        {"state": "x", "questions": {"bad id!": {"type": "noul", "instructions": "x"}}},
    ],
)
def test_text_schema_violations_are_400(client, auth, payload, backends):
    assert_error(client.post(TEXT, json=payload, headers=auth), 400, "invalid_request")
    assert backends.requests == []


def test_malformed_json_is_400(client, auth):
    resp = client.post(TEXT, content=b"{not json", headers={**auth, "Content-Type": "application/json"})
    assert_error(resp, 400, "invalid_request")


# ---------------------------------------------------------------- image


def test_image_success_wraps_backend(client, auth, image_request, siglip_response, backends):
    resp = client.post(IMAGE, json=image_request, headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "siglip2-base-patch16-224"
    assert body["model_version"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert body["results"] == siglip_response["results"]
    assert body["meta"]["backend_latency_ms"] == 64.5
    assert body["meta"]["hypothesis_template"] == "a photo of a {}"
    assert isinstance(body["latency_ms"], (int, float))
    (sent,) = backends.requests
    assert str(sent.url) == f"{IMAGE_URL}/classify"
    assert json.loads(sent.content) == image_request


def test_image_default_template_reported_in_meta(client, auth, image_request):
    del image_request["hypothesis_template"]
    body = client.post(IMAGE, json=image_request, headers=auth).json()
    assert body["meta"]["hypothesis_template"] == "a photo of a {}"


def test_image_more_than_64_labels_is_400(client, auth, image_request, backends):
    image_request["candidate_labels"] = [f"label {i}" for i in range(65)]
    body = assert_error(client.post(IMAGE, json=image_request, headers=auth), 400, "invalid_request")
    assert "64" in body["error"]["message"]
    assert backends.requests == []


def test_image_over_8_mib_decoded_is_413(client, auth, image_request, backends):
    image_request["image"]["base64"] = "A" * 11184816  # decodes to 8388612 bytes
    assert_error(client.post(IMAGE, json=image_request, headers=auth), 413, "payload_too_large")
    assert backends.requests == []


def test_image_body_over_12_mib_is_413(client, auth, image_request, backends):
    image_request["image"]["base64"] = "A" * (12 * 1024 * 1024 + 4)
    assert_error(client.post(IMAGE, json=image_request, headers=auth), 413, "payload_too_large")
    assert backends.requests == []


def test_image_invalid_base64_is_400(client, auth, image_request, backends):
    image_request["image"]["base64"] = "not*base64!"
    assert_error(client.post(IMAGE, json=image_request, headers=auth), 400, "invalid_request")
    assert backends.requests == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("candidate_labels"),
        lambda r: r.__setitem__("candidate_labels", ["dog", "dog"]),
        lambda r: r.__setitem__("hypothesis_template", "no placeholder"),
        lambda r: r.__setitem__("top_k", 0),
        lambda r: r["image"].__setitem__("media_type", "image/gif"),
        lambda r: r.__setitem__("unknown", True),
    ],
)
def test_image_schema_violations_are_400(client, auth, image_request, mutate, backends):
    mutate(image_request)
    assert_error(client.post(IMAGE, json=image_request, headers=auth), 400, "invalid_request")
    assert backends.requests == []


# ---------------------------------------------------------------- auth


@pytest.mark.parametrize("path,method", [(TEXT, "post"), (IMAGE, "post"), ("/v1/models", "get")])
def test_missing_key_is_401(client, path, method):
    resp = getattr(client, method)(path)
    body = assert_error(resp, 401, "unauthorized")
    assert resp.headers["WWW-Authenticate"].startswith("Bearer")
    assert API_KEY not in json.dumps(body)


@pytest.mark.parametrize(
    "header",
    [f"Bearer {API_KEY}x", "Bearer ", f"Basic {API_KEY}", API_KEY, f"bearer{API_KEY}"],
)
def test_invalid_key_is_401(client, laya_request, header, backends):
    resp = client.post(TEXT, json=laya_request, headers={"Authorization": header})
    assert_error(resp, 401, "unauthorized")
    assert "WWW-Authenticate" in resp.headers
    assert backends.requests == []


def test_any_configured_key_is_accepted(client, laya_request):
    resp = client.post(TEXT, json=laya_request, headers={"Authorization": f"Bearer {OTHER_KEY}"})
    assert resp.status_code == 200


def test_auth_checked_before_body_limit(client):
    resp = client.post(TEXT, content=b"x" * (100 * 1024), headers={"Content-Type": "application/json"})
    assert_error(resp, 401, "unauthorized")


# ---------------------------------------------------------------- backend errors


def raise_(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    "route,attr,payload_fixture",
    [(TEXT, "text", "laya_request"), (IMAGE, "image", "image_request")],
)
class TestBackendErrors:
    def call(self, client, auth, route, request, payload_fixture):
        return client.post(route, json=request.getfixturevalue(payload_fixture), headers=auth)

    def test_connect_error_is_503(self, client, auth, backends, route, attr, payload_fixture, request):
        setattr(backends, attr, raise_(httpx.ConnectError("connection refused")))
        body = assert_error(self.call(client, auth, route, request, payload_fixture), 503, "backend_unavailable")
        assert "unreachable" in body["error"]["message"]

    def test_timeout_is_504(self, client, auth, backends, route, attr, payload_fixture, request):
        setattr(backends, attr, raise_(httpx.ReadTimeout("timed out")))
        body = assert_error(self.call(client, auth, route, request, payload_fixture), 504, "backend_timeout")
        assert "10000 ms" in body["error"]["message"]

    def test_backend_500_is_502(self, client, auth, backends, route, attr, payload_fixture, request):
        setattr(backends, attr, lambda req: httpx.Response(500, text="Traceback: boom"))
        body = assert_error(self.call(client, auth, route, request, payload_fixture), 502, "backend_error")
        assert "HTTP 500" in body["error"]["message"]
        assert "Traceback" not in body["error"]["message"]
        assert body["latency_ms"] >= 0

    def test_backend_4xx_is_400_with_backend_message(
        self, client, auth, backends, route, attr, payload_fixture, request
    ):
        setattr(backends, attr, lambda req: httpx.Response(422, json={"detail": "unknown model 'klingon'"}))
        body = assert_error(self.call(client, auth, route, request, payload_fixture), 400, "invalid_request")
        assert "unknown model 'klingon'" in body["error"]["message"]

    def test_unparseable_backend_response_is_502(self, client, auth, backends, route, attr, payload_fixture, request):
        setattr(backends, attr, lambda req: httpx.Response(200, text="<html>"))
        assert_error(self.call(client, auth, route, request, payload_fixture), 502, "backend_error")

    def test_unexpected_error_is_500(self, client, auth, backends, route, attr, payload_fixture, request):
        setattr(backends, attr, raise_(RuntimeError("secret internal detail")))
        body = assert_error(self.call(client, auth, route, request, payload_fixture), 500, "internal_error")
        assert "secret internal detail" not in body["error"]["message"]


def test_backend_4xx_validation_list_detail(client, auth, backends, laya_request):
    backends.text = lambda req: httpx.Response(
        422, json={"detail": [{"loc": ["body", "questions"], "msg": "field required"}]}
    )
    body = assert_error(client.post(TEXT, json=laya_request, headers=auth), 400, "invalid_request")
    assert "field required" in body["error"]["message"]


# ---------------------------------------------------------------- health


def test_health_ok_without_auth(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backends"]["text"]["status"] == "ok"
    assert body["backends"]["image"]["status"] == "ok"
    assert body["backends"]["text"]["latency_ms"] >= 0


def test_health_degraded_when_one_down(client, backends):
    backends.image_health = raise_(httpx.ConnectError("refused"))
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["backends"] == {
        "text": {"status": "ok", "latency_ms": body["backends"]["text"]["latency_ms"]},
        "image": {"status": "down"},
    }


def test_health_degraded_when_text_returns_500(client, backends):
    backends.text_health = lambda req: httpx.Response(500)
    body = client.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert body["backends"]["text"] == {"status": "down"}


def test_health_down_is_503(client, backends):
    backends.text_health = raise_(httpx.ConnectTimeout("slow"))
    backends.image_health = raise_(httpx.ConnectError("refused"))
    resp = client.get("/v1/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "down", "backends": {"text": {"status": "down"}, "image": {"status": "down"}}}


def test_health_does_not_forward_authorization(client, auth, backends):
    client.get("/v1/health", headers=auth)
    assert all("authorization" not in {k.lower() for k in r.headers.keys()} for r in backends.requests)


# ---------------------------------------------------------------- models


def test_models_lists_contract_models(client, auth):
    resp = client.get("/v1/models", headers=auth)
    assert resp.status_code == 200
    assert resp.json() == {
        "models": [
            {"id": "laya-english", "task": "text-decision", "backend": "laya", "version": "laya-serve-0.4.20"},
            {"id": "laya-multilingual", "task": "text-decision", "backend": "laya", "version": "laya-serve-0.4.20"},
            {"id": "laya-typed-decisions", "task": "text-decision", "backend": "laya", "version": "laya-serve-0.4.20"},
            {
                "id": "siglip2-base-patch16-224",
                "task": "image-zero-shot",
                "backend": "siglip2",
                "version": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            },
        ]
    }


# ---------------------------------------------------------------- logging


def test_access_log_has_key_id_but_no_secrets(client, auth, laya_request, caplog):
    caplog.set_level(logging.INFO, logger="decision_api.access")
    client.post(TEXT, json=laya_request, headers=auth)
    records = [r for r in caplog.records if r.name == "decision_api.access"]
    assert len(records) == 1
    line = records[0].getMessage()
    entry = json.loads(line)
    assert entry["method"] == "POST"
    assert entry["path"] == TEXT
    assert entry["status"] == 200
    assert entry["latency_ms"] >= 0
    assert len(entry["key_id"]) == 8 and API_KEY[:8] not in entry["key_id"]
    full_log = "\n".join(r.getMessage() for r in caplog.records)
    assert API_KEY not in full_log
    assert "Duplicate charge" not in full_log


def test_access_log_for_unauthenticated_request(client, caplog):
    caplog.set_level(logging.INFO, logger="decision_api.access")
    client.get("/v1/models", headers={"Authorization": "Bearer wrong-key-value"})
    entry = json.loads([r for r in caplog.records if r.name == "decision_api.access"][0].getMessage())
    assert entry["status"] == 401
    assert entry["key_id"] is None
    assert "wrong-key-value" not in caplog.text


def test_openapi_docs_not_exposed(client):
    assert client.get("/docs").status_code in (401, 404)
    assert client.get("/openapi.json").status_code in (401, 404)
