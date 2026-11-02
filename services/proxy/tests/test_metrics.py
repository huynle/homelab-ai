"""Prometheus metrics: labels, histograms, backend gauge, separate metrics server (no network)."""

from __future__ import annotations

import time
import urllib.request

import httpx
import pytest
from fastapi.testclient import TestClient

import metrics
from app import create_app, key_id
from config import Settings
from conftest import API_KEY

REAL_START = metrics.start_metrics_server  # captured before the autouse fake replaces it

TEXT = "/v1/decide/text"
IMAGE = "/v1/classify/image"
KID = key_id(API_KEY)


def value(client, name: str, **labels) -> float | None:
    return client.app.state.metrics.registry.get_sample_value(name, labels)


def requests_total(client, route, model, code, kid=KID) -> float:
    return value(client, "decision_api_requests_total", route=route, model=model, code=str(code), key_id=kid) or 0.0


# ---------------------------------------------------------------- request counters


def test_success_counts_route_model_code_and_key_id(client, auth, laya_request):
    assert client.post(TEXT, json=laya_request, headers=auth).status_code == 200
    assert client.post(TEXT, json=laya_request, headers=auth).status_code == 200
    assert requests_total(client, TEXT, "laya-english", 200) == 2


def test_image_success_uses_registry_id(client, auth, image_request):
    assert client.post(IMAGE, json=image_request, headers=auth).status_code == 200
    assert requests_total(client, IMAGE, "siglip2-base-patch16-224", 200) == 1


def test_unauthorized_has_key_id_none_and_model_none(client, laya_request):
    assert client.post(TEXT, json=laya_request).status_code == 401
    assert requests_total(client, TEXT, "none", 401, kid="none") == 1


def test_invalid_request_400_counted_with_model_none(client, auth):
    assert client.post(TEXT, json={"state": "x"}, headers=auth).status_code == 400
    assert requests_total(client, TEXT, "none", 400) == 1


def test_unknown_model_400_has_model_none(client, auth, laya_request):
    resp = client.post(TEXT, json=dict(laya_request, model="does-not-exist"), headers=auth)
    assert resp.status_code == 400
    assert requests_total(client, TEXT, "none", 400) == 1


def test_backend_down_503_counts_request_and_backend_error(client, auth, laya_request, backends):
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    backends.text = down
    assert client.post(TEXT, json=laya_request, headers=auth).status_code == 503
    assert requests_total(client, TEXT, "laya-english", 503) == 1
    assert (
        value(client, "decision_api_backend_errors_total", model="laya-english", error_code="backend_unavailable") == 1
    )


def test_backend_error_502_counted(client, auth, image_request, backends):
    backends.image = lambda req: httpx.Response(500, text="boom")
    assert client.post(IMAGE, json=image_request, headers=auth).status_code == 502
    model = "siglip2-base-patch16-224"
    assert requests_total(client, IMAGE, model, 502) == 1
    assert value(client, "decision_api_backend_errors_total", model=model, error_code="backend_error") == 1


def test_backend_timeout_counted(client, auth, laya_request, backends):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    backends.text = slow
    assert client.post(TEXT, json=laya_request, headers=auth).status_code == 504
    assert value(client, "decision_api_backend_errors_total", model="laya-english", error_code="backend_timeout") == 1


def test_unknown_paths_collapse_to_other(client, auth):
    client.get("/v1/does-not-exist", headers=auth)
    client.get("/wp-admin/../etc/passwd", headers=auth)
    client.get("/random")  # unauthenticated
    assert requests_total(client, "other", "none", 404) == 2
    assert requests_total(client, "other", "none", 401, kid="none") == 1
    samples = client.app.state.metrics.registry.collect()
    routes = {s.labels["route"] for fam in samples for s in fam.samples if s.name == "decision_api_requests_total"}
    assert routes == {"other"}


def test_health_and_models_routes_are_labelled(client, auth):
    client.get("/v1/health")
    client.get("/v1/models", headers=auth)
    assert requests_total(client, "/v1/health", "none", 200, kid="none") == 1
    assert requests_total(client, "/v1/models", "none", 200) == 1


def test_healthz_is_excluded_from_request_metrics(client):
    assert client.get("/healthz").status_code == 200
    assert (
        value(client, "decision_api_requests_total", route="/healthz", model="none", code="200", key_id="none") is None
    )
    count = value(client, "decision_api_request_duration_seconds_count", route="/healthz", model="none")
    assert count is None


def test_labels_never_contain_raw_key_or_body(client, auth, laya_request):
    client.post(TEXT, json=laya_request, headers=auth)
    client.post(TEXT, json=laya_request, headers={"Authorization": "Bearer wrong-key-123"})
    text = metrics.generate_latest(client.app.state.metrics.registry).decode()
    assert API_KEY not in text
    assert "wrong-key-123" not in text
    for question in laya_request["questions"]:
        assert question not in text


# ---------------------------------------------------------------- histograms


def test_request_duration_histogram_observed_without_key_id(client, auth, laya_request):
    client.post(TEXT, json=laya_request, headers=auth)
    assert value(client, "decision_api_request_duration_seconds_count", route=TEXT, model="laya-english") == 1
    assert value(client, "decision_api_request_duration_seconds_bucket", route=TEXT, model="laya-english", le="10.0")
    families = {f.name: f for f in client.app.state.metrics.registry.collect()}
    for s in families["decision_api_request_duration_seconds"].samples:
        assert "key_id" not in s.labels


def test_request_duration_buckets_span_10ms_to_10s(client, auth, laya_request):
    client.post(TEXT, json=laya_request, headers=auth)
    families = {f.name: f for f in client.app.state.metrics.registry.collect()}
    les = {s.labels["le"] for s in families["decision_api_request_duration_seconds"].samples if "le" in s.labels}
    assert {"0.01", "10.0", "+Inf"} <= les


def test_backend_duration_histogram_observed(client, auth, laya_request, image_request):
    client.post(TEXT, json=laya_request, headers=auth)
    client.post(IMAGE, json=image_request, headers=auth)
    assert value(client, "decision_api_backend_request_duration_seconds_count", model="laya-english") == 1
    assert value(client, "decision_api_backend_request_duration_seconds_count", model="siglip2-base-patch16-224") == 1


def test_backend_duration_observed_on_failure(client, auth, laya_request, backends):
    backends.text = lambda req: httpx.Response(500)
    client.post(TEXT, json=laya_request, headers=auth)
    assert value(client, "decision_api_backend_request_duration_seconds_count", model="laya-english") == 1


# ---------------------------------------------------------------- backend_up and build_info


def test_backend_up_follows_health_probe(client, backends):
    client.get("/v1/health")
    assert value(client, "decision_api_backend_up", model="laya-english") == 1
    assert value(client, "decision_api_backend_up", model="laya-multilingual") == 1
    assert value(client, "decision_api_backend_up", model="siglip2-base-patch16-224") == 1

    backends.image_health = lambda req: httpx.Response(503)
    client.get("/v1/health")
    assert value(client, "decision_api_backend_up", model="siglip2-base-patch16-224") == 0
    assert value(client, "decision_api_backend_up", model="laya-english") == 1

    backends.image_health = lambda req: httpx.Response(200, json={"status": "ok"})
    client.get("/v1/health")
    assert value(client, "decision_api_backend_up", model="siglip2-base-patch16-224") == 1


def test_build_info(client):
    ids = "laya-english,laya-multilingual,laya-typed-decisions,siglip2-base-patch16-224"
    assert value(client, "decision_api_build_info", version=client.app.version, registry_models=ids) == 1


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_periodic_probe_updates_backend_up_without_traffic(settings, backends):
    s = Settings(**{**settings.__dict__, "backend_probe_interval_s": 0.02})
    app = create_app(s, transport=httpx.MockTransport(backends.handler))
    with TestClient(app) as c:
        assert _wait_for(lambda: value(c, "decision_api_backend_up", model="laya-english") == 1)
        backends.text_health = lambda req: httpx.Response(500)
        assert _wait_for(lambda: value(c, "decision_api_backend_up", model="laya-english") == 0)
    assert value(c, "decision_api_requests_total", route="/v1/health", model="none", code="200", key_id="none") is None


def test_periodic_probe_disabled_makes_no_backend_calls(settings, backends):
    s = Settings(**{**settings.__dict__, "backend_probe_interval_s": 0})
    app = create_app(s, transport=httpx.MockTransport(backends.handler))
    with TestClient(app):
        time.sleep(0.1)
    assert backends.requests == []


def test_periodic_probe_waits_one_interval_before_first_probe(client, backends):
    # default interval (30 s): nothing is probed during a short-lived client, so request counts stay exact
    time.sleep(0.05)
    assert backends.requests == []


# ---------------------------------------------------------------- metrics server


def test_main_app_does_not_serve_metrics(client, auth):
    for headers in ({}, auth):
        resp = client.get("/metrics", headers=headers)
        assert resp.status_code in (401, 404)
        assert "decision_api_" not in resp.text


def test_main_app_without_auth_does_not_serve_metrics(backends):
    app = create_app(Settings(api_keys=(), allow_no_auth=True), transport=httpx.MockTransport(backends.handler))
    with TestClient(app) as c:
        resp = c.get("/metrics")
    assert resp.status_code == 404
    assert "decision_api_" not in resp.text


def test_metrics_server_started_on_metrics_host_and_port(settings, backends, metrics_servers):
    s = Settings(**{**settings.__dict__, "host": "0.0.0.0", "metrics_port": 9191})
    with TestClient(create_app(s, transport=httpx.MockTransport(backends.handler))):
        assert [(c["host"], c["port"]) for c in metrics_servers.calls] == [("0.0.0.0", 9191)]
        assert metrics_servers.servers[0].stopped is False
    assert metrics_servers.servers[0].stopped is True


def test_metrics_host_overrides_host(settings, backends, metrics_servers):
    s = Settings(**{**settings.__dict__, "host": "0.0.0.0", "metrics_host": "127.0.0.1"})
    with TestClient(create_app(s, transport=httpx.MockTransport(backends.handler))):
        pass
    assert [(c["host"], c["port"]) for c in metrics_servers.calls] == [("127.0.0.1", 9090)]


def test_metrics_disabled_starts_no_server(settings, backends, metrics_servers):
    s = Settings(**{**settings.__dict__, "metrics_enabled": False})
    with TestClient(create_app(s, transport=httpx.MockTransport(backends.handler))):
        pass
    assert metrics_servers.calls == []


def test_metrics_server_bind_failure_does_not_stop_the_api(settings, backends, laya_request, auth, monkeypatch):
    def fail(**kwargs):
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(metrics, "start_metrics_server", fail)
    with TestClient(create_app(settings, transport=httpx.MockTransport(backends.handler))) as c:
        assert c.post(TEXT, json=laya_request, headers=auth).status_code == 200


def test_real_metrics_server_serves_registry(client, auth, laya_request):
    client.post(TEXT, json=laya_request, headers=auth)
    server = REAL_START(host="127.0.0.1", port=0, registry=client.app.state.metrics.registry)
    try:
        port = server.server_port
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:  # noqa: S310
            body = resp.read().decode()
    finally:
        server.shutdown()
        server.server_close()
    assert 'decision_api_requests_total{code="200",key_id="' + KID + '",model="laya-english"' in body
    assert "decision_api_build_info" in body


def test_registries_are_per_app(settings, backends):
    a = create_app(settings, transport=httpx.MockTransport(backends.handler))
    b = create_app(settings, transport=httpx.MockTransport(backends.handler))
    assert a.state.metrics.registry is not b.state.metrics.registry


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, (True, None, 9090, 30.0)),
        (
            {"METRICS_ENABLED": "false", "METRICS_HOST": "10.0.0.1", "METRICS_PORT": "9100"},
            (False, "10.0.0.1", 9100, 30.0),
        ),
        ({"METRICS_ENABLED": "1", "BACKEND_PROBE_INTERVAL_S": "0"}, (True, None, 9090, 0.0)),
    ],
)
def test_metrics_settings_from_env(env, expected):
    s = Settings.from_env(env)
    assert (s.metrics_enabled, s.metrics_host, s.metrics_port, s.backend_probe_interval_s) == expected
