# homelab-ai

Homelab AI services. The main one is the **decision API**: a small authenticated HTTP API on the ml VM
(`10.1.10.10`) for typed text decisions (Laya) and zero-shot image classification (SigLIP2).

| Route | Backend | Purpose |
|---|---|---|
| `POST /v1/decide/text` | Laya text server | `choice`, `score`, and `noul` (yes/no/unknown) answers about a text `state` |
| `POST /v1/classify/image` | SigLIP2 `siglip2-base-patch16-224` | pick among `candidate_labels` for an image |
| `GET /v1/health` | both | readiness (no auth) |
| `GET /v1/models` | both | served models |

Everything binds to localhost on ml; clients reach it through an SSH tunnel.

## Docs

- [`api/decision-api.yaml`](api/decision-api.yaml): OpenAPI 3.1 contract
- [`api/README.md`](api/README.md): envelope, error codes, limits, fixtures
- [`docs/runbook.md`](docs/runbook.md): start/stop, deploy, API keys, health checks, troubleshooting
- [`services/proxy/README.md`](services/proxy/README.md): the proxy (auth, limits, configuration)
- [`services/image/README.md`](services/image/README.md): the SigLIP2 image backend

## Usage

Open a tunnel from the laptop and keep it running:

```bash
ssh -i ~/.ssh/id_vms -N -L 8080:127.0.0.1:8080 huy@10.1.10.10
curl -s http://127.0.0.1:8080/v1/health
```

Every route except `/v1/health` needs `Authorization: Bearer <key>`. Get a key as described in the
[runbook](docs/runbook.md#api-key-management) and put it in `DECISION_API_KEY` (for example from a
password manager); do not hard-code it in scripts.

### Text decision

```bash
curl -s http://127.0.0.1:8080/v1/decide/text \
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

curl -s http://127.0.0.1:8080/v1/classify/image \
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
are limited to 8 MiB after decoding. See the [runbook's known caveats](docs/runbook.md#known-caveats).

Brain project: `homelab-ai`.
