"""Validate admin JSON responses against api/admin-api.yaml so the admin routes and the contract can't drift."""

from __future__ import annotations

import pytest
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

import admin as admin_mod
import keyservice as ks
from conftest import ALICE, API_DIR, BOB, OWNER

SPEC_URI = "urn:admin-api"


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load((API_DIR / "admin-api.yaml").read_text())


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
        validate({"error": {"code": "payload_too_large", "message": "x"}}, "ErrorEnvelope")
    with pytest.raises(AssertionError):
        validate({"email": "a@example.com", "role": "admin", "csrf_token": "x" * 40}, "Me")


def test_error_codes_match_contract(spec):
    codes = spec["components"]["schemas"]["Error"]["properties"]["code"]["enum"]
    assert set(codes) == set(admin_mod.STATUS)
    for code, status in ks.STATUS.items():
        assert admin_mod.STATUS[code] == status


def test_contract_request_examples_are_accepted(admin, spec):
    owner = admin.signed_in(OWNER)
    paths = spec["paths"]
    for example in paths["/api/keys"]["post"]["requestBody"]["content"]["application/json"]["examples"].values():
        assert admin.call(owner, "POST", "/admin/api/keys", example["value"]).status_code == 201
    for example in paths["/api/members"]["post"]["requestBody"]["content"]["application/json"]["examples"].values():
        assert admin.call(owner, "POST", "/admin/api/members", example["value"]).status_code == 201


def test_success_responses_match_schemas(admin, validate, laya_request):
    ks.invite_member(admin.db, ks.get_viewer(admin.db, OWNER), ALICE, now=admin.clock())
    owner, alice = admin.signed_in(OWNER), admin.signed_in(ALICE)
    validate(owner.get("/admin/api/me").json(), "Me")
    validate(alice.get("/admin/api/me").json(), "Me")

    created = admin.call(alice, "POST", "/admin/api/keys", {"name": "ha", "expires_at": "2027-11-02T00:00:00Z"})
    validate(created.json(), "CreatedKey")
    key = created.json()
    validate(owner.get("/admin/api/keys").json(), "KeyList")
    validate(alice.get("/admin/api/keys").json(), "KeyList")
    validate(admin.call(alice, "PATCH", f"/admin/api/keys/{key['id']}", {"name": "ha2"}).json(), "Key")

    api = admin.client()
    auth = {"Authorization": f"Bearer {key['key']}"}
    api.post("/v1/decide/text", json=laya_request, headers=auth)
    api.post("/v1/decide/text", json={"state": "x"}, headers=auth)  # 400 before a model is resolved
    api.get("/v1/models", headers=auth)
    assert admin.app.state.usage.flush()
    for group_by in ("day", "key", "model", "route"):
        for c in (owner, alice):
            report = c.get("/admin/api/usage", params={"group_by": group_by})
            assert report.status_code == 200
            validate(report.json(), "UsageReport")
            assert report.json()["rows"], group_by
            dims = {"day": {"date"}, "key": {"key_id", "key_name", "owner_email"}, "model": {"model"},
                    "route": {"route"}}[group_by]  # fmt: skip
            totals = {"requests", "errors", "client_errors", "server_errors", "total_latency_ms", "avg_latency_ms"}
            assert all(set(r) == dims | totals for r in report.json()["rows"])

    validate(owner.get("/admin/api/members").json(), "MemberList")
    invited = admin.call(owner, "POST", "/admin/api/members", {"email": BOB})
    validate(invited.json(), "Member")
    validate(admin.call(alice, "POST", f"/admin/api/keys/{key['id']}/revoke").json(), "Key")
    validate(admin.call(owner, "DELETE", f"/admin/api/members/{ALICE}").json(), "MemberRemoved")
    page = owner.get("/admin/api/audit").json()
    validate(page, "AuditPage")
    events = {e["event"] for e in page["entries"]}
    assert {"sign_in.allowed", "member.invited", "member.removed", "key.created", "key.revoked"} <= events
    admin.sign_in(admin.client(), "mallory@example.com")
    validate(owner.get("/admin/api/audit").json(), "AuditPage")


def test_error_responses_match_schemas(admin, validate):
    ks.invite_member(admin.db, ks.get_viewer(admin.db, OWNER), ALICE, now=admin.clock())
    alice = admin.signed_in(ALICE)
    cases = [
        (admin.client().get("/admin/api/me"), 401, "unauthorized"),
        (alice.get("/admin/api/members"), 403, "forbidden"),
        (admin.call(alice, "POST", "/admin/api/keys", {"name": "x"}, csrf=None), 403, "csrf_failed"),
        (admin.call(alice, "POST", "/admin/api/keys/aaaaaaaaaaaa/revoke"), 404, "not_found"),
        (admin.call(alice, "POST", "/admin/api/keys", {"name": "x", "rate_limit_rpm": 1}), 400, "invalid_request"),
        (admin.call(alice, "POST", "/admin/api/members", {"email": ALICE}), 403, "forbidden"),
    ]
    owner = admin.signed_in(OWNER)
    cases.append((admin.call(owner, "POST", "/admin/api/members", {"email": ALICE}), 409, "conflict"))
    for resp, status, code in cases:
        assert resp.status_code == status, resp.text
        validate(resp.json(), "ErrorEnvelope")
        assert resp.json()["error"]["code"] == code
        assert set(resp.json()) == {"error"}
