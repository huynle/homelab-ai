# Decision API proxy

FastAPI service that implements [`api/decision-api.yaml`](../../api/decision-api.yaml) in front of the
model backends listed in its [backend registry](#backend-registry):

| Route | Backend call |
|---|---|
| `POST /v1/decide/text` | `text-decision` model, Laya adapter: `POST {backend_url}/predict` |
| `POST /v1/classify/image` | `image-zero-shot` model, zero-shot-image adapter: `POST {backend_url}/classify` |
| `GET /v1/health` | `GET {backend_url}/health` once per distinct backend (public, no auth) |
| `GET /v1/models` | the registry entries |
| `GET /healthz` | none: process liveness for Kubernetes (public, no auth, not access-logged) |

Prometheus metrics are served on a separate, internal-only port (default `9090`); see [Metrics](#metrics).

## Production

Production runs this proxy on k3s behind `https://decide.ai.huynle.com` (namespace `decision-api`).
CI ([`.github/workflows/proxy-image.yml`](../../.github/workflows/proxy-image.yml)) publishes
`ghcr.io/huynle/decision-api-proxy` on every push to `main` that touches `services/proxy/**`, and Flux
image automation in `huynle/infra` (`apps/homelab/decision-api/`) promotes it. The registry and the API
keys (SOPS Secret `decision-api-keys`) live there too; a change to either needs
`kubectl -n decision-api rollout restart deploy/decision-api`. Operations: [runbook](../../docs/runbook.md).

## Files

- `app.py`: `create_app()` (routes, auth, body limits, model resolution, health, access log) and `main()`
- `registry.py`: the backend registry (YAML schema, validation, env fallback, model resolution)
- `adapters.py`: one class per backend type (`laya`, `zero-shot-image`): backend call, response mapping
  and backend error mapping
- `config.py`: `Settings.from_env()` and API key loading
- `keystore.py`: the key stores (`FileKeyStore`, `SqliteKeyStore`), key format, HMAC hashing; see
  [Key store](#key-store)
- `db.py`, `migrations/`: SQLite connection settings and forward-only numbered migrations (SQLite mode only)
- `keyservice.py`: the service layer of the [admin API](../../api/admin-api.yaml) (keys, members, usage and
  audit queries with the owner/member rules). No HTTP routes yet
- `usage.py`: the in-memory per-key daily usage aggregator and its flush to SQLite
- `metrics.py`: Prometheus metrics (per-app registry) and the separate metrics HTTP server
- `registry.example.yaml`: registry for the `ml` backends on the LAN
- `Dockerfile`: container image (non-root, listens on `0.0.0.0:8080`)
- `models.py`: Pydantic request models and limits that mirror the spec's component schemas
- `errors.py`: the `ApiError` code -> HTTP status table and the `ErrorEnvelope` response
- `tests/`: pytest suite. Backends are mocked with `httpx.MockTransport` using the real captures in
  `api/fixtures/`, so the tests make no network calls. `tests/test_contract.py` validates responses against
  the OpenAPI component schemas with `jsonschema`, so the proxy and the contract cannot drift apart.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `DECISION_API_KEYS_FILE` | unset | Path to the key file: one key per line, `#` starts a comment |
| `DECISION_API_KEYS` | unset | Comma-separated keys (added to those in the file) |
| `DECISION_API_ALLOW_NO_AUTH` | `0` | `1` lets the proxy start with no keys and serve without auth. **Tests only.** |
| `DECISION_API_REGISTRY_FILE` | unset | Path to the [backend registry](#backend-registry) YAML. When unset, the registry is built from the variables below |
| `HOST` | `127.0.0.1` | Bind address (the container image sets `0.0.0.0`) |
| `PORT` | `8080` | Bind port |
| `HEALTH_TIMEOUT_S` | `2` | Per-backend timeout for the `/v1/health` probe |
| `METRICS_ENABLED` | `1` | `0`/`false` starts no metrics server (metrics are still recorded in memory) |
| `METRICS_HOST` | same as `HOST` | Bind address of the metrics server |
| `METRICS_PORT` | `9090` | Port of the metrics server (`GET /metrics`) |
| `BACKEND_PROBE_INTERVAL_S` | `30` | Period of the background backend health probe that refreshes `decision_api_backend_up`; `0` disables it |
| `KEYSTORE` | `file` | `file`: keys from the variables above, nothing written. `sqlite`: the [SQLite key store](#key-store) (opt-in) |
| `DECISION_API_DB_PATH` | unset | SQLite mode: database file; its directory must exist and be writable |
| `DECISION_API_KEY_PEPPER_FILE` | unset | SQLite mode: file with the HMAC pepper, at least 32 bytes (surrounding whitespace is stripped) |
| `DECISION_API_OWNER_EMAILS` | unset | SQLite mode: comma-separated owner emails (lowercased); the first one owns imported keys |
| `DECISION_API_USAGE_FLUSH_S` | `15` | SQLite mode: seconds between usage flushes to the database (also flushed on shutdown) |

Used only when `DECISION_API_REGISTRY_FILE` is unset (the pre-registry setup, still the default):

| Variable | Default | Meaning |
|---|---|---|
| `TEXT_BACKEND_URL` | `http://127.0.0.1:8000` | Laya server (`laya-english`, `laya-multilingual`, `laya-typed-decisions`) |
| `IMAGE_BACKEND_URL` | `http://127.0.0.1:8010` | Image (SigLIP2) service |
| `TEXT_TIMEOUT_S` | `10` | Laya request timeout; when exceeded the proxy returns 504 `backend_timeout` |
| `IMAGE_TIMEOUT_S` | `10` | Image request timeout; when exceeded the proxy returns 504 `backend_timeout` |
| `LAYA_MODEL_VERSION` | `laya-serve-0.4.20` | `model_version` in text responses and `/v1/models` |
| `IMAGE_MODEL_ID` | `siglip2-base-patch16-224` | `model` in image responses and `/v1/models` |
| `IMAGE_MODEL_VERSION` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Fallback when the backend sends no `model_version` |

The proxy refuses to start when no key is configured (file mode), the registry is invalid, or, with
`KEYSTORE=sqlite`, the pepper is missing or shorter than 32 bytes, the database directory is missing or no
owner email is set; it prints one line saying why and exits with status 1.

## Backend registry

The registry lists every model the proxy serves. It is read once at startup from
`DECISION_API_REGISTRY_FILE`; see [`registry.example.yaml`](registry.example.yaml) for the `ml` backends.

```yaml
defaults: {text: laya-english, image: siglip2-base-patch16-224}
models:
  - id: laya-english               # what clients send as `model`; listed by /v1/models
    task: text-decision            # text-decision | image-zero-shot
    adapter: laya                  # laya | zero-shot-image
    backend_url: http://10.1.10.10:8000
    backend_model: english         # optional; forwarded to the backend (Laya `model`)
    version: laya-serve-0.4.20     # reported as model_version
    timeout_s: 10                  # optional, default 10
  - id: siglip2-base-patch16-224
    task: image-zero-shot
    adapter: zero-shot-image
    backend: siglip2               # optional label for /v1/models; defaults to the adapter key
    backend_url: http://10.1.10.10:8010
    version: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
```

Startup validation (any failure stops the proxy): ids are unique and match `[A-Za-z0-9_.-]{1,128}`,
adapters are known and serve the entry's `task`, `backend_url` is an `http(s)` URL without a query string,
`timeout_s` is in (0, 600], unknown keys are rejected, and both `defaults` exist and match their task.

Routing:

- A request's `model` selects the entry with that id. It must belong to the route's task; otherwise the
  proxy answers 400 `invalid_request` and lists the valid ids. Without `model`, `defaults.text` or
  `defaults.image` is used.
- Text requests may still name a Laya checkpoint (`english`, `multilingual`, `typed-decisions`, or a Laya
  alias such as `en`, `multi`, `typed`). It maps to the entry `laya-<checkpoint>` when that id exists.
- Laya adapter: `backend_model` is forwarded as Laya's `model` only when the client chose the model. A
  request without `model` goes to the default entry's backend unpinned, so Laya keeps routing by
  script/language, and the response reports `laya-<routing.model>` when that entry exists on the same
  backend.
- The image request's `model` is not forwarded to the image backend.

### Adding a model

1. Start the backend on `ml` and check `GET {backend_url}/health`.
2. Add an entry to the registry file. A model served by an existing adapter needs no code change.
3. Restart the proxy. In production the registry is infra `apps/homelab/decision-api/registry.yaml`;
   after merging, run `kubectl -n decision-api rollout restart deploy/decision-api` (see the
   [runbook](../../docs/runbook.md#add-a-model)). `GET /v1/models` lists it and
   `GET /v1/health` reports it under `models`.

A new kind of backend needs a new adapter: one class in `adapters.py` with a `task` and an async `run()`
that calls the backend and returns a `Result`, registered in `ADAPTERS` under the key used in `adapter:`.

## API keys

In production keys come from the SOPS Secret in infra (see the
[runbook](../../docs/runbook.md#api-key-management)). For a local run:

```bash
install -d -m 700 /etc/decision-api
umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' >> /etc/decision-api/keys
chmod 600 /etc/decision-api/keys   # the proxy logs a warning if group/others can read it
```

Add one key per caller, with a `# comment` line above it to name it. Keys must not contain `#`. To revoke a
key, delete its line and restart the proxy. Callers send `Authorization: Bearer <key>`. Keys are compared
in constant time (`hmac.compare_digest`) against every configured key.

## Key store

`KEYSTORE` selects where valid keys come from. **The default is `file` and stays the default** until the
deployment has a persistent volume for the database.

- **`file` (default).** The keys from `DECISION_API_KEYS_FILE` / `DECISION_API_KEYS`, compared in constant
  time against every key. No database is opened and nothing is written. Behaviour is the same as before the
  SQLite store; `DECISION_API_DB_PATH` and the other SQLite variables are ignored unless `KEYSTORE=sqlite`.
- **`sqlite` (opt-in).** Keys live in SQLite and are managed through the admin API (the HTTP routes and
  Google sign-in come later; `keyservice.py` is the service layer). At startup the proxy opens the
  database, applies pending migrations, makes the `DECISION_API_OWNER_EMAILS` owners, and imports every key
  from the key file as an owner key (`source: imported`, name `imported (SOPS)`, `imported (SOPS) 2`, ...,
  prefix `imported…`). The import is idempotent: a key already stored, even revoked, is skipped, so revoking
  the imported key sticks across restarts. The key file stays the break-glass source: switching back to
  `KEYSTORE=file` serves its keys again. Every request needs a key in this mode (`DECISION_API_ALLOW_NO_AUTH`
  is ignored).

Issued keys look like `dk_<public_id>_<secret>`: `public_id` is 12 lowercase base32 characters (`a-z`,
`2-7`) and is the key's id; `secret` is 32 random bytes, base64url without padding (43 characters). The
database stores only `HMAC-SHA256(pepper, key)` (hex), never the key; the full key is returned once, when it
is created. Authentication looks the key up by its public id and compares the HMAC with
`hmac.compare_digest`; imported keys (no public id in their value) are looked up by their HMAC. Revoked keys
and keys whose `expires_at` has passed fail with the same 401 as unknown keys. Active keys are cached in
memory and the cache is reloaded after every write to the database (the proxy runs one replica), so a
revoke takes effect on the next request.

Rules enforced by the service layer (see [`admin-api.yaml`](../../api/admin-api.yaml)): names 1-64
characters; members hold at most 20 active keys (the owner has no limit); `expires_at` is optional, in the
future and at most 730 days ahead; revoke is soft and idempotent (no second audit entry); removing a member
revokes their active keys and deletes their sessions in one transaction; members see only their own keys
and usage (another user's key is `not_found`, another `user` filter is `forbidden`); usage ranges are UTC
dates, both inclusive, default the last 30 days, at most 366 days.

**Usage.** For requests that presented a valid key, the proxy adds one count per (UTC day, key, route
template, model) in memory: requests, client errors (4xx), server errors (5xx) and total latency. The
counts go to `usage_daily` every `DECISION_API_USAGE_FLUSH_S` seconds and on shutdown; a failed flush keeps
its counts for the next one. `last_used_at` is written by the same flush, at most once a minute per key.
Requests without a resolved model are stored with model `''` and reported as `null`. In file mode no usage
is recorded (the Prometheus metrics are unchanged in both modes).

**Logs and metrics.** `key_id` in the access log and the `key_id` metric label is the public id for issued
keys and the 8-hex sha256 prefix for file and imported keys, as before.

**Schema** (`migrations/0001_initial.sql`): `users` (email, role `owner|member`, invited by/at, removed at,
last sign-in), `api_keys` (public id, unique key HMAC, name, owner, source `issued|imported`, prefix,
created/expires/revoked/last-used times, reserved `rate_limit_rpm`), `usage_daily` (primary key day, key,
route, model), `sessions` (hashed id and CSRF token, expiry) and `audit` (time, actor, event, target, key
id, JSON detail; no IP addresses or tokens). Connections use WAL, `foreign_keys=ON` and a 5 s busy timeout.

**Migrations are forward-only.** Files are `migrations/NNNN_name.sql`, numbered from `0001` without gaps,
and each runs once in its own transaction, recorded in `schema_version`. Never edit a migration that has
shipped: add the next number. There are no down migrations, and the proxy refuses to start on a database
whose schema is newer than its code (roll back by restoring a backup, not by downgrading).

**Backup.** Back the database up with SQLite's online backup API (for example `sqlite3 "$DB" ".backup
/backup/decision-api.db"`), which is safe while the proxy writes in WAL mode; this is what the k3s-backup
CronJob pattern does. Do not copy the database file alone: the `-wal` file holds recent commits. The pepper
is not in the database; without it, stored keys cannot be checked, so back it up with the other secrets.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
DECISION_API_KEYS_FILE=/etc/decision-api/keys .venv/bin/python app.py
# with a registry: DECISION_API_REGISTRY_FILE=registry.yaml DECISION_API_KEYS_FILE=... .venv/bin/python app.py
# or: DECISION_API_KEYS_FILE=... .venv/bin/uvicorn app:create_app --factory --host 127.0.0.1 --port 8080 --no-access-log
```

```bash
curl -s localhost:8080/v1/health
curl -s -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d @../../api/fixtures/laya-predict.request.json localhost:8080/v1/decide/text
```

### Container

The image runs as uid 10001, listens on `0.0.0.0:8080` (`HOST`/`PORT` are set in the image) and reads keys
and the registry from mounted files. Use `GET /healthz` for the liveness probe and `GET /v1/health` for
readiness.

```bash
docker build -t decision-api services/proxy
docker run --rm -p 127.0.0.1:8080:8080 \
  -v /etc/decision-api:/etc/decision-api:ro \
  -e DECISION_API_KEYS_FILE=/etc/decision-api/keys \
  -e DECISION_API_REGISTRY_FILE=/etc/decision-api/registry.yaml \
  decision-api
curl -s localhost:8080/healthz
```

The key file must be readable by uid 10001 inside the container (in Kubernetes, mount the Secret with
`defaultMode: 0400` and `fsGroup`/`runAsUser: 10001`).

## Behaviour

- **Text.** `state` and `questions` are forwarded to Laya unchanged, plus the entry's `backend_model` as
  `model` when the client selected a model (see [Routing](#backend-registry)); `min_confidence` stays in
  the proxy. The Laya response maps to the envelope as follows:
  - `answers` -> `results`
  - `routing` minus `repo` -> `meta.routing`
  - `usage` -> `meta.usage`
  - Laya's `model` -> `meta.backend_model`
  - `model` = the selected registry id, or `laya-<routing.model>` (for example `laya-english`) when Laya
    routed the request itself

  When `min_confidence` is sent, every answer gets `low_confidence`: true when `confidence < min_confidence`,
  false otherwise. An answer without a numeric `confidence` is flagged as low.
- **Image.** The request body goes to the backend unchanged, except for `model`, which the proxy consumes. In the response, `results` passes through, and
  the backend's `latency_ms` becomes `meta.backend_latency_ms`, next to `meta.hypothesis_template`,
  `meta.pad_to_square` (the effective value; default false),
  `meta.backend_model`, and any `meta`/`device` the backend returns. `latency_ms` is the proxy's total time.
- **Limits.** These checks run before any backend call:
  - Body cap before JSON parsing: 64 KiB for text and 12 MiB for image. Over the cap returns 413.
  - `state` over 32 KiB of UTF-8 JSON returns 413.
  - An image over 8 MiB decoded (or more than 11,184,812 base64 characters) returns 413.
  - More than 32 questions or more than 64 labels returns 400.
  - Schema violations and invalid base64 return 400.
- **Errors.** Every error uses `{"error": {"code", "message"}}` and includes `latency_ms`:

  | Failure | Response |
  |---|---|
  | Backend connect error | 503 `backend_unavailable` |
  | Backend timeout | 504 `backend_timeout` |
  | Backend 4xx | 400 `invalid_request`, carrying the backend's message |
  | Backend 5xx or unparseable reply | 502 `backend_error` |
  | Anything else | 500 `internal_error` (no details) |
- **Health.** Each distinct `backend_url` is probed once. `ok` when every backend answers `/health` with 200,
  `degraded` when some do (still HTTP 200), `down` when none does (HTTP 503). `backends.text` and
  `backends.image` report the default models' backends; `models` reports every model by id.
- **Liveness.** `GET /healthz` returns `{"status": "ok"}` without auth or backend calls, and is not written
  to the access log (kubelet probes would flood it).
- **Logging.** The proxy writes one JSON access line per request, with `method`, `path`, `status`,
  `latency_ms` and `key_id`. `key_id` is the first 8 hex characters of the key's sha256 (for issued
  `dk_` keys in SQLite mode, the key's public id). Keys, request
  bodies and query strings are never logged, and the client's `Authorization` header is never forwarded to
  the backends. Uvicorn's own access log is turned off. The FastAPI `/docs` and `/openapi.json` endpoints
  are disabled; the YAML file is the contract.

## Metrics

The proxy serves Prometheus metrics with `prometheus_client`'s own HTTP server, started in the app lifespan,
on `METRICS_HOST:METRICS_PORT` (default `HOST:9090`; in the image `0.0.0.0:9090`). Scrape path: `/metrics`.
The port is **internal only**: it is not exposed by the ingress (which routes only to the API port 8080) and
is meant to be scraped in-cluster. The public API port never serves `/metrics` (it answers 401 without a
key and 404 with one). If the metrics port cannot be bound, the proxy logs an error and keeps serving the API.
The proxy runs as a single uvicorn process, so no multiprocess mode is used.

| Metric | Type | Labels |
|---|---|---|
| `decision_api_requests_total` | counter | `route`, `model`, `code`, `key_id` |
| `decision_api_request_duration_seconds` | histogram | `route`, `model` |
| `decision_api_backend_request_duration_seconds` | histogram | `model` |
| `decision_api_backend_errors_total` | counter | `model`, `error_code` |
| `decision_api_backend_up` | gauge | `model` |
| `decision_api_build_info` | gauge (always 1) | `version`, `registry_models` |

Labels:

- `route`: the route template (`/v1/decide/text`, `/v1/classify/image`, `/v1/health`, `/v1/models`).
  Any other path is `other`, so scanners cannot create new series.
- `model`: the registry id that served the request (for Laya without `model`, the checkpoint it routed to),
  or `none` when no model was resolved (401, 400 before model resolution, health, models).
- `code`: the HTTP status. `key_id`: the same value as the access log (8-hex sha256 prefix, or the public id
  of an issued key), `none` without a valid key. Raw keys, emails and request bodies never appear in labels.
- `error_code`: the contract error code of a failed backend call (`backend_unavailable`, `backend_timeout`,
  `backend_error`, `invalid_request` for a backend 4xx, `internal_error`).
- `version`: the proxy/contract version (`1.1.0`); `registry_models`: comma-separated registry ids.

Durations use buckets from 10 ms to 10 s (`0.01 ... 10`). `backend_request_duration_seconds` includes failed
calls (a timeout is observed at its timeout). `GET /healthz` is excluded from request metrics, as it is
from the access log. `decision_api_backend_up` is 1/0 per model from the last health probe of its backend:
every `GET /v1/health` (including readiness probes) updates it, and a background probe every
`BACKEND_PROBE_INTERVAL_S` keeps it fresh without traffic (the first background probe runs one interval after
start). The standard `process_*` and `python_*` collectors are included.

## Tests

```bash
python3 -m venv /tmp/proxy-venv && /tmp/proxy-venv/bin/pip install -r requirements-dev.txt
cd services/proxy && /tmp/proxy-venv/bin/python -m pytest -q
ruff check . && ruff format --check .
```
