# Decision API v1

`decision-api.yaml` (OpenAPI 3.1.0) is the contract for a small proxy on the `ml` VM. It sits in
front of two local backends:

| Route | Backend | Task |
|---|---|---|
| `POST /v1/decide/text` | Laya (`laya[serve]` 0.4.20, `127.0.0.1:8000/predict`) | typed text decisions: `choice`, `score`, `noul` |
| `POST /v1/classify/image` | SigLIP2 `google/siglip2-base-patch16-224` @ `75de2d55` | zero-shot image classification |
| `GET /v1/health` | both | readiness (public, no auth) |
| `GET /v1/models` | both | served models |

Every route except `GET /v1/health` requires `Authorization: Bearer <api-key>`.

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

Failures return `{"error": {"code", "message"}}`. The `model` fields appear only once a model has
been resolved. Each code maps to one HTTP status:

| code | HTTP |
|---|---|
| `invalid_request` | 400 |
| `unauthorized` | 401 |
| `payload_too_large` | 413 |
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

## Later

An OmniRoute route may later map onto this contract, so callers could reach it through the same
gateway as the LLM providers. The contract does not depend on that.
