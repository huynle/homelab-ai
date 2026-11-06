"""Validate proxy responses against the OpenAPI component schemas so proxy and contract can't drift."""

from __future__ import annotations

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from app import create_app
from config import Settings
from conftest import API_DIR, API_KEY

PROXY_DIR = API_DIR.parent / "services" / "proxy"

SPEC_URI = "urn:decision-api"


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load((API_DIR / "decision-api.yaml").read_text())


@pytest.fixture(scope="module")
def validate(spec):
    registry = Registry().with_resource(SPEC_URI, Resource.from_contents(spec, default_specification=DRAFT202012))

    def _validate(instance, schema_name: str):
        schema = {"$ref": f"{SPEC_URI}#/components/schemas/{schema_name}"}
        errors = list(Draft202012Validator(schema, registry=registry).iter_errors(instance))
        assert not errors, [f"{list(e.absolute_path)}: {e.message}" for e in errors]

    return _validate


def test_validator_actually_rejects(validate):
    with pytest.raises(AssertionError):
        validate({"error": {"code": "nope", "message": "x"}}, "ErrorEnvelope")


def test_text_envelope_matches_schema(client, auth, laya_request, validate):
    validate(laya_request, "TextDecisionRequest")
    resp = client.post("/v1/decide/text", json=dict(laya_request, min_confidence=0.5), headers=auth)
    assert resp.status_code == 200
    validate(resp.json(), "TextDecisionEnvelope")


def test_image_envelope_matches_schema(client, auth, image_request, validate):
    validate(image_request, "ImageClassifyRequest")
    resp = client.post("/v1/classify/image", json=image_request, headers=auth)
    assert resp.status_code == 200
    validate(resp.json(), "ImageClassifyEnvelope")


def test_health_matches_schema(client, backends, validate):
    validate(client.get("/v1/health").json(), "HealthResponse")
    backends.text_health = lambda req: httpx.Response(503)
    backends.image_health = lambda req: httpx.Response(503)
    validate(client.get("/v1/health").json(), "HealthResponse")


def test_models_match_schema_and_spec_example(client, auth, spec, validate):
    body = client.get("/v1/models", headers=auth).json()
    validate(body, "ModelList")
    example = spec["paths"]["/v1/models"]["get"]["responses"]["200"]["content"]["application/json"]["examples"]
    assert body == example["current"]["value"]


def test_errors_match_schema(client, auth, backends, laya_request, validate):
    responses = [
        client.get("/v1/models"),
        client.post("/v1/decide/text", json={"state": "x"}, headers=auth),
        client.post("/v1/decide/text", json=dict(laya_request, state="x" * 40000), headers=auth),
    ]
    backends.text = lambda req: httpx.Response(500)
    responses.append(client.post("/v1/decide/text", json=laya_request, headers=auth))
    for resp in responses:
        assert resp.status_code >= 400
        validate(resp.json(), "ErrorEnvelope")


def test_spec_request_examples_are_accepted(spec, backends, settings):
    """Every request example in the spec gets past the proxy's own validation."""
    app = create_app(settings, transport=httpx.MockTransport(backends.handler))
    headers = {"Authorization": f"Bearer {settings.api_keys[0]}"}
    with TestClient(app) as c:
        for path in ("/v1/decide/text", "/v1/classify/image"):
            examples = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["examples"]
            for ex in examples.values():
                # the image example's base64 is elided ("...") - make it valid while keeping its prefix
                body = ex["value"]
                if "image" in body:
                    body = dict(body, image=dict(body["image"], base64="/9j/4AAQSkZJRgABAQAAAQABAAD/"))
                assert c.post(path, json=body, headers=headers).status_code == 200, (path, body)


def test_example_registry_models_match_spec_example(spec, backends, validate):
    settings = Settings(api_keys=(API_KEY,), registry_file=str(PROXY_DIR / "registry.example.yaml"))
    app = create_app(settings, transport=httpx.MockTransport(backends.handler))
    with TestClient(app) as c:
        body = c.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}).json()
    validate(body, "ModelList")
    example = spec["paths"]["/v1/models"]["get"]["responses"]["200"]["content"]["application/json"]["examples"]
    assert body == example["example_registry"]["value"]


def test_image_request_may_name_a_model(image_request, validate):
    validate(dict(image_request, model="siglip2-base-patch16-224"), "ImageClassifyRequest")


def test_health_models_map_matches_schema(client, backends, spec, validate):
    assert "models" in spec["components"]["schemas"]["HealthResponse"]["properties"]
    body = client.get("/v1/health").json()
    assert set(body["models"]) == {
        "laya-english",
        "laya-multilingual",
        "laya-typed-decisions",
        "siglip2-base-patch16-224",
    }
    validate(body, "HealthResponse")
    with pytest.raises(AssertionError):
        validate({"status": "ok", "backends": body["backends"], "models": {"x": {"status": "bogus"}}}, "HealthResponse")


def test_healthz_matches_schema(client, spec, validate):
    op = spec["paths"]["/healthz"]["get"]
    assert op["security"] == []
    resp = client.get("/healthz")
    assert resp.status_code == 200
    validate(resp.json(), "Liveness")


# ---------------------------------------------------------------- /v1/decisions (OpenAI wire format)

PNG_URL = "data:image/png;base64," + "iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def _decision_examples(spec):
    op = spec["paths"]["/v1/decisions"]["post"]
    req = op["requestBody"]["content"]["application/json"]["examples"]["support_ticket"]["value"]
    resp = op["responses"]["200"]["content"]["application/json"]["examples"]["support_ticket"]["value"]
    return req, resp


def test_decision_spec_example_round_trips_through_the_proxy(client, auth, spec, validate):
    """The spec's request example, answered by the real Laya capture, gives exactly the spec's response example."""
    req, resp_example = _decision_examples(spec)
    validate(req, "DecisionCreateRequest")
    validate(resp_example, "Decision")
    resp = client.post("/v1/decisions", json=req, headers=auth)
    assert resp.status_code == 200, resp.text
    validate(resp.json(), "Decision")
    assert resp.json() == resp_example


def test_decision_image_and_refusal_answers_match_schema(client, auth, backends, validate):
    image_req = {
        "model": "gpt-6-luna",
        "input": [{"role": "user", "content": [{"type": "input_image", "image_url": PNG_URL}]}],
        "questions": [
            {"type": "choice", "instructions": "?", "choices": [{"value": "german shepherd"}, {"value": "dog"}]}
        ],
    }
    validate(image_req, "DecisionCreateRequest")
    resp = client.post("/v1/decisions", json=image_req, headers=auth)
    assert resp.status_code == 200, resp.text
    validate(resp.json(), "Decision")

    backends.text = lambda req: httpx.Response(200, json={"answers": {}, "usage": {"input_tokens": 3}})
    text_req = {"input": "x", "questions": [{"type": "predicate", "instructions": "?"}]}
    validate(text_req, "DecisionCreateRequest")
    resp = client.post("/v1/decisions", json=text_req, headers=auth)
    assert resp.json()["answers"] == [{"type": "refusal", "name": None}]
    validate(resp.json(), "Decision")


def test_decision_errors_match_schema(client, auth, validate):
    for resp in (
        client.post("/v1/decisions", json={"input": "x", "questions": []}, headers=auth),
        client.post("/v1/decisions", json={"input": "x", "questions": [{"type": "predicate", "instructions": "?"}]}),
    ):
        assert resp.status_code >= 400
        validate(resp.json(), "ErrorEnvelope")


def test_schema_rejects_what_the_proxy_rejects(validate):
    with pytest.raises(AssertionError):
        validate({"input": "x", "questions": [{"type": "box", "instructions": "?"}]}, "DecisionCreateRequest")
    with pytest.raises(AssertionError):
        validate(
            {"input": "x", "questions": [{"type": "predicate", "instructions": "?"}], "stream": True},
            "DecisionCreateRequest",
        )
    with pytest.raises(AssertionError):
        bad_image = [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}]
        validate(
            {"input": bad_image, "questions": [{"type": "predicate", "instructions": "?"}]}, "DecisionCreateRequest"
        )
