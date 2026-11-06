# Decision API v1

`decision-api.yaml` (OpenAPI 3.1.0, version 1.5.0) is the contract for a small proxy served at
`https://decide.ai.huynle.com` (k3s). It sits in front of two backends on the `ml` VM (`10.1.10.10`):

| Route | Backend | Task |
|---|---|---|
| `POST /v1/decide/text` | Laya (`laya[serve]` 0.4.20, `ml:8000/predict`) | typed text decisions: `choice`, `score`, `noul` |
| `POST /v1/classify/image` | SigLIP2 `google/siglip2-base-patch16-224` @ `75de2d55` | zero-shot image classification |
| `POST /v1/decisions` | Laya, or SigLIP2 for one image | [OpenAI Decisions API](https://developers.openai.com/api/reference/resources/decisions/methods/create) wire format |
| `POST /api/alpha/decisions` | Laya, SigLIP2, or OpenRouter (paid) | [OpenRouter Decisions](https://openrouter.ai/docs/guides/community/multimodal-decisions) wire format |
| `GET /v1/health` | every backend | readiness (public, no auth) |
| `GET /v1/models` | registry | served models |
| `GET /healthz` | none | proxy liveness (public, no auth) |

Every `/v1` route except `GET /v1/health`, and `POST /api/alpha/decisions`, require `Authorization: Bearer <api-key>`; `GET /healthz` is
public too. Missing, unknown, expired and revoked keys all get the same 401 `unauthorized`. `/admin` is not
part of this contract; it has its own sign-in and its own spec (see [Admin API](#admin-api)).

The proxy reads the served models from a backend registry (`services/proxy/registry.example.yaml`). A
request picks one with `model` (an id from `GET /v1/models`); without it, the registry's default model for
the task is used. Text requests may still send a Laya checkpoint name such as `english`.

## Borrowed conventions

- **Laya question format.** `state` plus `questions: {id: {type, instructions, criteria}}` is passed
  to Laya unchanged, so Laya docs and payloads work as they are. Answer fields keep Laya's names
  (`choice`, `score`, `noul`, `probabilities`, `confidence`, `answer_confidence`, `legend`, `action`).
- **Hugging Face zero-shot naming.** `candidate_labels`, `hypothesis_template` (default
  `"a photo of a {}"`), `top_k`, and results as `[{label, score}]` sorted by descending score, as in
  the HF `zero-shot-image-classification` pipeline.
- **KServe V2-style ops.** Readiness is rolled up per backend in `/v1/health`, and `/v1/models` is a
  flat model listing with `id`, `task`, `backend`, and `version`.
- **OpenAPI 3.1.** JSON Schema 2020-12 dialect: `const`, discriminated `oneOf` for question and
  answer types, and `contentEncoding: base64`.

## OpenAI Decisions API

`POST /v1/decisions` follows OpenAI's Decisions wire format instead of the envelope below: `model`,
`input` (a string or an array of messages, tool calls and tool outputs) and `questions`
(`[{type: predicate|choice|score, name, instructions, choices|levels}]`) in; `{model, answers, usage}`
out, with answers in question order. The OpenAI SDK works with `base_url` set to
`https://decide.ai.huynle.com/v1`.

- `gpt-6-luna` means the default model; registry ids and Laya checkpoint names pin a model.
- Text, JSON and conversations go to Laya (`predicate` -> `noul`). Requests with more than 32
  questions are split across several Laya calls.
- Images, text wins: one base64 PNG or JPEG image with text or JSON evidence goes to Laya on that
  evidence alone; the image is ignored and the response carries `X-Decision-Image: ignored`. Only an
  image with nothing else goes to SigLIP2, with `choice` questions only (one zero-shot classification
  per question over its choice values). Hosted image URLs, more than one image and images in question
  instructions return 400.
- Limits: 400 questions, 64 choices, 10 levels, 2048-character instructions, 32 KiB of text input.
- `prompt_cache_*` and `safety_identifier` are accepted and ignored; `usage` cache and reasoning
  counts are 0.

## OpenRouter Decisions API

`POST /api/alpha/decisions` speaks OpenRouter's Decisions wire format at OpenRouter's own path, so an
OpenRouter SDK works by changing only its base URL to `https://decide.ai.huynle.com`. In: `model`
(required), `state` (string, object or array), `questions` keyed by id (`noul`, `choice`, `score`, `box`),
and optional `assets`, `provider`, `session_id`, `trace`, `user`. Out: `{model, answers, usage, id,
provider}` (plus `annotations` when OpenRouter sends them). The same route serves local models and, for
keys allowed to, paid OpenRouter models such as TypeSafe Jev.

- **Models and aliases.** `model` is a registry id or alias from `GET /v1/models` (or a Laya checkpoint
  name such as `english`). The registry's `aliases` map names to models: `fast: laya-english` always uses
  that model; `auto: {text: laya-english, image: jev}` uses `image` when the request has `image_url`,
  `input_audio` or `video_url` parts, `assets` or box questions, and `text` otherwise. The response's
  `model` is the registry id that served the request, never the upstream versioned id. Registry models
  with task `decisions` (OpenRouter) and aliases are reachable only here, never through the `/v1` routes.
- **Capabilities.** Each model lists what it accepts (`text`, `image`, `box`, `audio`, `video`). Laya is
  text only. SigLIP2 takes exactly one image as the whole state (an `image_url` part, an asset key, or the
  only item of a state array) and answers `choice` questions over the option keys. OpenRouter models list
  theirs in the registry. A request the model cannot serve is 400 before any backend call (for example
  `model 'laya-english' does not accept images`).
- **Paid models.** Models served by OpenRouter (`paid: true` in `/v1/models`) need a key whose owner
  granted paid access in the admin page; other keys get 403 `forbidden` before anything is sent upstream.
  Without an OpenRouter key on the proxy, paid models answer 503 `backend_unavailable`. OpenRouter's
  `usage.cost` is returned and summed per key in the admin usage report (`cost_usd`).
- **Forwarded vs dropped.** OpenRouter gets `state`, `questions`, `assets` and `provider` exactly as sent,
  with `model` replaced by the upstream model id (for example `typesafe/jev-1.14`); images, box questions
  and assets are not translated. `session_id`, `trace` and `user` are accepted but never forwarded.
- **Local translation (Laya).** `state` strings and objects pass through; an array of strings is joined
  with newlines and any other array becomes compact JSON. Instructions and criteria that are objects or
  arrays become compact JSON strings; a `null` choice criterion becomes the option name. Laya's `noul`
  takes no criteria, so noul `true`/`false` criteria are appended to the instructions as `True if: ...`
  and `False if: ...`. Missing instructions become `Answer this question about the state.` Answers map
  back to `{type: noul, noul}`, `{type: choice, choice, probabilities, confidence}` and `{type: score,
  score, probabilities, legend, confidence}` (`legend` echoes each criterion as the client sent it); a
  question Laya did not answer is `{type: refusal}`. Local responses have `provider: homelab`,
  `usage.cost: 0` and an id like `gen-dec-<unix seconds>-<20 alphanumerics>`.
- **Limits.** Body 16 MiB; at most 400 questions; `session_id` and `user` at most 256 characters.
  Images are base64 `data:` URLs (PNG, JPEG or WebP; local models PNG or JPEG only, 8 MiB decoded) or
  `assets` (`{format: png|jpg|jpeg, data_base64}`); every asset must be referenced in `state`,
  instructions or criteria (as the whole string or inside text), else 400 naming the key. Hosted
  `http(s)` image URLs are 400 (OpenRouter does not fetch them either). Local models also need a
  non-empty state of at most 32 KiB (413), 2-64 choice options, 2-16 score levels and instructions of at
  most 2048 characters; more than 32 questions are split across Laya calls.
- **Errors.** This route uses OpenRouter's envelope, including for 401 and 413:
  `{"error": {"code": <HTTP status>, "message": "...", "metadata": {"code": "<contract code>", "param":
  <field or null>}}}`. Upstream 400/422 become 400 and 413 stays 413 (with OpenRouter's message), 429
  becomes 429 `rate_limited`, 401/402/403 become 502 `backend_error` with a generic message (the proxy's
  OpenRouter credentials or credits), 503 stays 503, other 5xx are 502 and timeouts 504.

## Envelope

Success responses use one shape:

```json
{"model": "laya-english", "model_version": "laya-serve-0.4.20", "latency_ms": 575,
 "results": { "...task-specific..." }, "meta": {"routing": {}, "usage": {}}}
```

- `results` is `qid -> answer` for text and `[{label, score}]` for images.
- `meta` carries backend details. For text these are Laya's `routing` (which checkpoint served the
  request and why), `usage`, and Laya's own `model` name as `meta.backend_model`. The proxy removes
  `routing.repo` because it is a local filesystem path.
  For images, `meta.hypothesis_template` and `meta.pad_to_square` report the effective values,
  including defaults (`pad_to_square` is off unless the client sends `true`).

Failures return `{"error": {"code", "message", "type", "param"}}` (on `/api/alpha/*`, OpenRouter's envelope
described above); `type` and `param` follow OpenAI's
error object (`invalid_request_error`, `authentication_error` or `server_error`, and the offending field or null).
A backend's own 4xx message is passed on with filesystem paths replaced by `[path]`; a backend reply that
contains NaN or Infinity is a 502 `backend_error`. The `model` fields appear only once a model has
been resolved. Each code maps to one HTTP status:

| code | HTTP |
|---|---|
| `invalid_request` | 400 |
| `unauthorized` | 401 |
| `forbidden` | 403 |
| `payload_too_large` | 413 |
| `rate_limited` | 429 |
| `internal_error` | 500 |
| `backend_error` | 502 |
| `backend_unavailable` | 503 |
| `backend_timeout` | 504 |

## Limits

- Text `state`: at most 32 KiB once serialized as UTF-8 JSON (exceeding it returns 413). A request
  may have at most 32 questions (exceeding it returns 400).
- Image: at most 8 MiB after base64 decoding (at most 11,184,812 base64 characters; exceeding it
  returns 413). A request may have at most 64 `candidate_labels`.

## Fixtures

`fixtures/` holds real captures from `ml`, not invented data:

- `laya-predict*.json`, `laya-health.json`, `laya-models.json`: raw responses from the running Laya
  server on 2026-11-01. `*.request.json` are the matching request payloads. The batch payload uses
  Laya's `states` array; v1 of this API does not expose batching.
- `siglip-*-results.json`: SigLIP2 benchmark outputs. They store per-label probabilities as `prob`
  (softmax over labels of `logit_scale * cosine`). The API renames this to `score`.

The examples in the spec are trimmed versions of these files.

## Admin API

`admin-api.yaml` (OpenAPI 3.1.0, version 1.1.0, separate from the public contract) describes the admin page at
`https://decide.ai.huynle.com/admin`: sign-in, API keys, usage and audit. The design is
[ADR 0033](https://github.com/huynle/infra/blob/main/docs/decisions/0033-decision-api-admin-google-sign-in-sqlite-keys.md)
in `huynle/infra`.

- **Sign-in.** The app runs its own Google OIDC sign-in (PKCE, nonce, `email_verified` required) with the
  redirect URI `https://decide.ai.huynle.com/admin/auth/callback`. It never trusts `X-Forwarded-User`.
  The session is a server-side session in the `__Host-decision_session` cookie (`HttpOnly`, `Secure`,
  `SameSite=Lax`, 12 hours). `POST`, `PATCH` and `DELETE` also need the `X-CSRF-Token` header (from
  `GET /admin/api/me`) and a same-origin `Origin` header.
- **Roles.** The owner's email comes from config. The owner invites members by email, sees every key and
  all usage, and reads the audit log. Members manage only their own keys and see only their own usage.
  Removing a member revokes all of their active keys and ends their sessions; the keys and their usage
  stay visible to the owner.
- **Google Testing mode.** The OAuth consent screen is in Testing mode, so an in-app invite is not enough:
  the owner must also add the invitee as a Google test user in GCP (project `huynle-infra`, at most 100).
  Otherwise Google refuses the sign-in.
- **Keys.** Format `dk_<public_id>_<secret>`: a 12-character lowercase base32 public id (also the key's
  `id`) and a 32-byte base64url secret. Keys are stored as HMAC-SHA256 with a server pepper. The full key
  is shown once, at creation; listings show the prefix `dk_<public_id>_…`, name, owner email, created,
  expiry, revoked and last-used times. Revoke is soft (`revoked_at`) and takes effect at once. Names are
  1-64 characters, members may hold 20 active keys (the owner has no limit), and `expires_at` is optional,
  in the future and at most 2 years ahead. `rate_limit_rpm` is reserved: always `null` and ignored for now.
- **Usage.** Per-key daily totals in UTC by route and model: requests, client errors (4xx), server
  errors (5xx), total latency (for the average) and upstream cost (`cost_usd`, OpenRouter's
  `usage.cost` summed; 0 for local models).
- **Paid models.** Keys have `allow_paid` (default false). Only the owner changes it
  (`PATCH /admin/api/keys/{id}` with `{"allow_paid": true}`, or the toggle on the keys page); it applies
  on the key's next request and is audited as `key.paid_access_changed`. Kept indefinitely; no raw request logs. Query with
  `from`, `to`, `key_id`, `user` and `group_by=day|key|model|route`.
- **Audit.** Owner-only log of sign-ins allowed and denied, members invited and removed, keys created
  and revoked, and paid access changed.
- **Legacy key.** The single key from the SOPS file is imported once as an owner key named
  `imported (SOPS)`. The SOPS file stays as a break-glass key source.

Admin errors use the same `{"error": {"code", "message"}}` envelope with the codes `invalid_request` (400),
`unauthorized` (401), `forbidden` (403), `csrf_failed` (403), `not_found` (404), `conflict` (409) and
`internal_error` (500).

## Later

An OmniRoute route may later map onto this contract, so callers could reach it through the same
gateway as the LLM providers. The contract does not depend on that.
