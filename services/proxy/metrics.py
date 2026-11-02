"""Prometheus metrics for the decision proxy.

Every app instance owns its own :class:`~prometheus_client.CollectorRegistry` (no global registry, so tests and
multiple apps never collide). The registry is served by prometheus_client's own HTTP server on a separate port
(``METRICS_PORT``, default 9090); the public API port never exposes ``/metrics``.

Labels are bounded and never carry secrets or request content: ``route`` is a route template (unknown paths
collapse to ``other``), ``model`` is a registry id or ``none``, ``key_id`` is the 8-hex sha256 prefix also used
in the access log (or ``none``), ``code`` is the HTTP status.
"""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

__all__ = ["Metrics", "generate_latest", "start_metrics_server"]

PREFIX = "decision_api"
NONE = "none"
OTHER_ROUTE = "other"

# 10 ms .. 10 s (the default backend timeout), roughly 1-2.5-5 steps
REQUEST_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    """All proxy metrics, registered on one per-app registry."""

    def __init__(self, routes: Iterable[str], version: str, model_ids: Iterable[str]):
        self.registry = CollectorRegistry(auto_describe=True)
        self.routes = frozenset(routes)
        reg = self.registry
        ProcessCollector(registry=reg)
        PlatformCollector(registry=reg)
        GCCollector(registry=reg)

        self.requests = Counter(
            f"{PREFIX}_requests_total",
            "HTTP requests handled by the proxy (excluding /healthz)",
            ("route", "model", "code", "key_id"),
            registry=reg,
        )
        self.request_duration = Histogram(
            f"{PREFIX}_request_duration_seconds",
            "End-to-end request latency in the proxy (excluding /healthz)",
            ("route", "model"),
            buckets=REQUEST_BUCKETS,
            registry=reg,
        )
        self.backend_duration = Histogram(
            f"{PREFIX}_backend_request_duration_seconds",
            "Latency of backend inference calls, including failed ones",
            ("model",),
            buckets=REQUEST_BUCKETS,
            registry=reg,
        )
        self.backend_errors = Counter(
            f"{PREFIX}_backend_errors_total",
            "Failed backend inference calls by contract error code",
            ("model", "error_code"),
            registry=reg,
        )
        self.backend_up = Gauge(
            f"{PREFIX}_backend_up",
            "1 when the model's backend answered its last health probe, else 0",
            ("model",),
            registry=reg,
        )
        self.build_info = Gauge(
            f"{PREFIX}_build_info",
            "Proxy version and the registry model ids; always 1",
            ("version", "registry_models"),
            registry=reg,
        )
        self.build_info.labels(version=version, registry_models=",".join(model_ids)).set(1)

    def route(self, path: str) -> str:
        return path if path in self.routes else OTHER_ROUTE

    def observe_request(self, path: str, model: str | None, code: int, key_id: str | None, seconds: float) -> None:
        route = self.route(path)
        model = model or NONE
        self.requests.labels(route=route, model=model, code=str(code), key_id=key_id or NONE).inc()
        self.request_duration.labels(route=route, model=model).observe(seconds)

    def observe_backend(self, model: str, seconds: float, error_code: str | None = None) -> None:
        self.backend_duration.labels(model=model).observe(seconds)
        if error_code is not None:
            self.backend_errors.labels(model=model, error_code=error_code).inc()

    def set_backend_up(self, model: str, up: bool) -> None:
        self.backend_up.labels(model=model).set(1 if up else 0)


def start_metrics_server(*, host: str, port: int, registry: CollectorRegistry):
    """Serve ``registry`` on ``host:port`` in a daemon thread; returns the server (``shutdown()`` to stop)."""
    server, _thread = start_http_server(port, addr=host, registry=registry)
    return server
