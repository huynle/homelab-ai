"""Per-key daily usage, aggregated in memory and flushed to SQLite (``usage_daily`` and ``last_used_at``).

Only requests that presented a valid key are recorded (others cannot be attributed). Rows are keyed by
(UTC day, key public id, route template, model); ``model`` None means no model was resolved. A failed flush
puts its counts back, so a transient database error loses nothing; the next flush retries.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections.abc import Callable
from datetime import datetime

from db import UTC, Database, iso, upsert_usage, utcnow

log = logging.getLogger("decision_api")

Key = tuple[str, str, str, str | None]  # day, public_id, route, model


class UsageAggregator:
    def __init__(self, db: Database, now: Callable[[], datetime] = utcnow, last_used_interval_s: float = 60.0):
        self._db = db
        self._now = now
        self._interval = last_used_interval_s
        self._lock = threading.Lock()
        self._counts: dict[Key, list] = {}  # [requests, client_errors, server_errors, latency_ms_sum, cost_usd]
        self._seen: dict[str, datetime] = {}  # public_id -> latest use not yet written to last_used_at
        self._written: dict[str, datetime] = {}  # public_id -> last_used_at last written

    def record(
        self, public_id: str, route: str, model: str | None, status: int, latency_ms: float, cost_usd: float = 0.0
    ) -> None:
        """Count one request. ``cost_usd``: the upstream (OpenRouter) cost; non-finite or negative counts as 0."""
        if not isinstance(cost_usd, (int, float)) or not math.isfinite(cost_usd) or cost_usd < 0:
            cost_usd = 0.0
        at = self._now()
        key = (at.astimezone(UTC).date().isoformat(), public_id, route, model)
        with self._lock:
            c = self._counts.get(key)
            if c is None:
                c = self._counts[key] = [0, 0, 0, 0.0, 0.0]
            c[0] += 1
            if 400 <= status < 500:
                c[1] += 1
            elif status >= 500:
                c[2] += 1
            c[3] += latency_ms
            c[4] += cost_usd
            if public_id not in self._seen or self._seen[public_id] < at:
                self._seen[public_id] = at

    def pending(self) -> dict[Key, tuple]:
        with self._lock:
            return {k: tuple(v) for k, v in self._counts.items()}

    def _restore(self, counts: dict[Key, list], seen: dict[str, datetime]) -> None:
        with self._lock:
            for key, add in counts.items():
                c = self._counts.setdefault(key, [0, 0, 0, 0.0, 0.0])
                for i, v in enumerate(add):
                    c[i] += v
            for public_id, at in seen.items():
                if public_id not in self._seen or self._seen[public_id] < at:
                    self._seen[public_id] = at

    def flush(self) -> bool:
        """Write pending counts; True on success (or nothing to do), False when the write failed."""
        with self._lock:
            counts, self._counts = self._counts, {}
            seen, self._seen = self._seen, {}
        touch, deferred = {}, {}
        for public_id, at in seen.items():
            last = self._written.get(public_id)
            if last is None or (at - last).total_seconds() >= self._interval:
                touch[public_id] = at
            else:
                deferred[public_id] = at  # written by a later flush, at most once a minute
        if deferred:
            self._restore({}, deferred)
        if not counts and not touch:
            return True
        try:
            with self._db.write(bump=False) as conn:
                for (day, public_id, route, model), (req, ce, se, lat, cost) in counts.items():
                    upsert_usage(conn, day, public_id, route, model, req, ce, se, lat, cost)
                for public_id, at in touch.items():
                    ts = iso(at)
                    conn.execute(
                        "UPDATE api_keys SET last_used_at = ?"
                        " WHERE public_id = ? AND (last_used_at IS NULL OR last_used_at < ?)",
                        (ts, public_id, ts),
                    )
        except Exception as exc:  # noqa: BLE001 (keep the counts whatever failed)
            log.warning("usage flush failed (%s: %s); keeping the counts for the next flush", type(exc).__name__, exc)
            self._restore(counts, touch)
            return False
        self._written.update(touch)
        return True

    async def run(self, interval_s: float) -> None:
        """Flush every ``interval_s`` seconds until cancelled (the lifespan flushes once more on shutdown)."""
        while True:
            await asyncio.sleep(interval_s)
            await asyncio.to_thread(self.flush)
