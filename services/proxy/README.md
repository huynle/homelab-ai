# Decision API proxy

FastAPI service that implements [`api/decision-api.yaml`](../../api/decision-api.yaml) in front of two
local backends:

| Route | Backend call |
|---|---|
| `POST /v1/decide/text` | Laya `POST {TEXT_BACKEND_URL}/predict` |
| `POST /v1/classify/image` | image service `POST {IMAGE_BACKEND_URL}/classify` |
| `GET /v1/health` | `GET {backend}/health` on both (public, no auth) |
| `GET /v1/models` | static list from config |

## Files

- `app.py`: `create_app()` (routes, auth, body limits, error mapping, access log) and `main()`
- `config.py`: `Settings.from_env()` and API key loading
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
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8080` | Bind port |
| `TEXT_BACKEND_URL` | `http://127.0.0.1:8000` | Laya server |
| `IMAGE_BACKEND_URL` | `http://127.0.0.1:8010` | Image (SigLIP2) service |
| `TEXT_TIMEOUT_S` | `10` | Laya request timeout; when exceeded the proxy returns 504 `backend_timeout` |
| `IMAGE_TIMEOUT_S` | `10` | Image request timeout; when exceeded the proxy returns 504 `backend_timeout` |
| `HEALTH_TIMEOUT_S` | `2` | Per-backend timeout for the `/v1/health` probe |
| `LAYA_MODEL_VERSION` | `laya-serve-0.4.20` | `model_version` in text responses and `/v1/models` |
| `IMAGE_MODEL_ID` | `siglip2-base-patch16-224` | `model` in image responses and `/v1/models` |
| `IMAGE_MODEL_VERSION` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Fallback when the backend sends no `model_version` |

The proxy refuses to start when no key is configured.

## API keys

```bash
install -d -m 700 /etc/decision-api
umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' >> /etc/decision-api/keys
chmod 600 /etc/decision-api/keys   # the proxy logs a warning if group/others can read it
```

Add one key per caller, with a `# comment` line above it to name it. Keys must not contain `#`. To revoke a
key, delete its line and restart the proxy. Callers send `Authorization: Bearer <key>`. Keys are compared
in constant time (`hmac.compare_digest`) against every configured key.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
DECISION_API_KEYS_FILE=/etc/decision-api/keys .venv/bin/python app.py
# or: DECISION_API_KEYS_FILE=... .venv/bin/uvicorn app:create_app --factory --host 127.0.0.1 --port 8080 --no-access-log
```

```bash
curl -s localhost:8080/v1/health
curl -s -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d @../../api/fixtures/laya-predict.request.json localhost:8080/v1/decide/text
```

## Behaviour

- **Text.** `state`, `questions` and `model` (when given) are forwarded to Laya unchanged; `min_confidence`
  stays in the proxy. The Laya response maps to the envelope as follows:
  - `answers` -> `results`
  - `routing` minus `repo` -> `meta.routing`
  - `usage` -> `meta.usage`
  - Laya's `model` -> `meta.backend_model`
  - `model` = `laya-<routing.model>`, for example `laya-english`

  When `min_confidence` is sent, every answer gets `low_confidence`: true when `confidence < min_confidence`,
  false otherwise. An answer without a numeric `confidence` is flagged as low.
- **Image.** The request body goes to the backend unchanged. In the response, `results` passes through, and
  the backend's `latency_ms` becomes `meta.backend_latency_ms`, next to `meta.hypothesis_template`,
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
- **Health.** `ok` when both backends answer `/health` with 200, `degraded` when one does (still HTTP 200),
  `down` when neither does (HTTP 503).
- **Logging.** The proxy writes one JSON access line per request, with `method`, `path`, `status`,
  `latency_ms` and `key_id`. `key_id` is the first 8 hex characters of the key's sha256. Keys, request
  bodies and query strings are never logged, and the client's `Authorization` header is never forwarded to
  the backends. Uvicorn's own access log is turned off. The FastAPI `/docs` and `/openapi.json` endpoints
  are disabled; the YAML file is the contract.

## Tests

```bash
python3 -m venv /tmp/proxy-venv && /tmp/proxy-venv/bin/pip install -r requirements-dev.txt
cd services/proxy && /tmp/proxy-venv/bin/python -m pytest -q
ruff check . && ruff format --check .
```
