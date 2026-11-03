# homelab-ai

Homelab AI services. The main one is the **decision API**: a small authenticated HTTP API for typed
text decisions (Laya) and zero-shot image classification (SigLIP2), served at
**https://decide.ai.huynle.com**.

| Route | Backend | Purpose |
|---|---|---|
| `POST /v1/decide/text` | Laya text server | `choice`, `score`, and `noul` (yes/no/unknown) answers about a text `state` |
| `POST /v1/classify/image` | SigLIP2 `siglip2-base-patch16-224` | pick among `candidate_labels` for an image |
| `POST /v1/decisions` | Laya, or SigLIP2 for one image | [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions) wire format (`predicate`, `choice`, `score`) |
| `GET /v1/health` | both | readiness (no auth) |
| `GET /v1/models` | both | served models |
| `GET /healthz` | none | proxy liveness (no auth) |

## Architecture

```text
client --HTTPS--> Traefik (k3s, wildcard TLS, rate limit) --> proxy pod (namespace decision-api)
                                                                 |  API keys, limits, registry
                                                                 v
                                        ml VM 10.1.10.10: laya :8000, decision-image :8010 (GPU)
```

- **Proxy** ([`services/proxy`](services/proxy)): container image `ghcr.io/huynle/decision-api-proxy`,
  built by CI on every push to `main` and promoted to k3s automatically by Flux image automation.
- **Backends** ([`services/laya`](services/laya), [`services/image`](services/image)): systemd units on the
  `ml` VM, pinned to the Tesla P40. They have no auth; the firewall on ml admits only the k3s node.
- **Operations** live in the private `huynle/infra` repo: `apps/homelab/decision-api/` (HelmRelease,
  ingress, backend registry, SOPS-encrypted API keys) and `hosts/ml/decision-api/` (systemd units, env
  files, `deploy.sh`, `firewall.sh`). See the [runbook](docs/runbook.md).

## Docs

- [`api/decision-api.yaml`](api/decision-api.yaml): OpenAPI 3.1 contract
- [`api/README.md`](api/README.md): envelope, error codes, limits, fixtures
- [`docs/runbook.md`](docs/runbook.md): topology, deploys, adding models, API keys, health checks, troubleshooting, rollback
- [`services/proxy/README.md`](services/proxy/README.md): the proxy (auth, limits, configuration)
- [`services/image/README.md`](services/image/README.md): the SigLIP2 image backend

## Usage

```bash
export DECISION_API=https://decide.ai.huynle.com
curl -s $DECISION_API/v1/health
```

Every route except `/v1/health` and `/healthz` needs `Authorization: Bearer <key>`. Get a key as
described in the [runbook](docs/runbook.md#api-key-management) and put it in `DECISION_API_KEY` (for
example from a password manager); do not hard-code it in scripts.

On the LAN the name resolves through Pi-hole to the k3s node (`10.1.10.30`), so the same URL works at
home and outside.

### Text decision

```bash
curl -s $DECISION_API/v1/decide/text \
  -H "Authorization: Bearer $DECISION_API_KEY" \
  -H 'content-type: application/json' \
  -d '{
    "state": {"subject": "Duplicate charge on invoice #4411",
              "body": "We were billed twice for March. Please refund it today or we will cancel."},
    "questions": {
      "department": {"type": "choice", "instructions": "Which department should handle this request?",
                     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                  "sales": "pricing, new contracts", "other": "everything else"}},
      "urgency": {"type": "score", "instructions": "How urgent is this request?",
                  "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
      "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
    }
  }'
```

`results` maps each question id to its answer. The full payload is
[`api/fixtures/laya-predict.request.json`](api/fixtures/laya-predict.request.json), with the
matching response in [`api/fixtures/laya-predict.json`](api/fixtures/laya-predict.json).

### Image classification

The image goes in the JSON body as standard base64. With `jq`:

```bash
jq -n --rawfile b64 <(base64 -i photo.jpg | tr -d '\n') \
  '{image: {base64: $b64, media_type: "image/jpeg"},
    candidate_labels: ["dog", "cat", "receipt"],
    top_k: 2}' > /tmp/classify.json

curl -s $DECISION_API/v1/classify/image \
  -H "Authorization: Bearer $DECISION_API_KEY" \
  -H 'content-type: application/json' \
  -d @/tmp/classify.json
```

Or build the body with Python:

```bash
python3 -c '
import base64, json, sys
b64 = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({"image": {"base64": b64, "media_type": "image/jpeg"},
                  "candidate_labels": ["dog", "cat", "receipt"], "top_k": 2}))
' photo.jpg > /tmp/classify.json
```

The response's `results` is `[{label, score}]`, highest score first. `pad_to_square` is off by default;
padding makes scores for tall document images such as receipts less decisive, so send
`"pad_to_square": true` only for images where squashing distorts the subject. Images
are limited to 8 MiB after decoding.

### OpenAI Decisions API

`POST /v1/decisions` accepts OpenAI's Decisions request body and returns its `Decision` object, so the
OpenAI SDK works by pointing `base_url` at the proxy. `gpt-6-luna` selects the default model:

```python
import os

from openai import OpenAI

client = OpenAI(base_url="https://decide.ai.huynle.com/v1", api_key=os.environ["DECISION_API_KEY"])
decision = client.decisions.create(
    model="gpt-6-luna",
    input="I was charged twice for my order.",
    questions=[{"type": "choice", "name": "department", "instructions": "Which department?",
                "choices": [{"value": "billing"}, {"value": "technical"}, {"value": "other"}]}],
)
print(decision.answers[0].choice)
```

Text, JSON and conversations are answered by Laya. A request with one base64 image and only `choice`
questions is answered by SigLIP2; other image requests and hosted image URLs return 400. See the
[contract](api/decision-api.yaml) (`createDecision`) for the mapping and limits.

Typical latency through the public route from the LAN: text decisions about 0.16–0.26 s, image
classification of a photo about 0.35 s. See the [runbook's known caveats](docs/runbook.md#known-caveats).

Brain project: `homelab-ai`.
