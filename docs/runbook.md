# Decision API runbook

Operations guide for the decision API at **https://decide.ai.huynle.com**. The contract is
[`api/decision-api.yaml`](../api/decision-api.yaml); see [`api/README.md`](../api/README.md) for the
envelope, error codes, and limits.

This repo holds the code. Deployment config lives in the private `huynle/infra` repo:

| infra path | What |
|---|---|
| `apps/homelab/decision-api/release.yaml` | Namespace, HelmRelease (proxy Deployment, Service, NetworkPolicy, Traefik ingress) |
| `apps/homelab/decision-api/registry.yaml` | Backend registry, generated as ConfigMap `decision-api-registry` |
| `apps/homelab/decision-api/decision-api-keys.sops.yaml` | API keys (SOPS-encrypted Secret `decision-api-keys`) |
| (to add) an OpenRouter key Secret | The OpenRouter API key for paid models, mounted as a file for `DECISION_API_OPENROUTER_KEY_FILE` (see [Paid OpenRouter models](#paid-openrouter-models)) |
| `apps/homelab/decision-api/image-policy.yaml` | Flux image repository/policy for the proxy image |
| `hosts/ml/decision-api/` | systemd units, env files, `deploy.sh`, `firewall.sh` for the backends on ml |
| `docs/decisions/0030-decision-api-public-api-key-route.md` | Why the API is public behind API keys |

## Topology

```text
client --HTTPS--> Traefik on k3s (10.1.10.30) --> decision-api pod --> ml 10.1.10.10 :8000 (laya)
                  wildcard TLS, public-chain       API keys, limits,       :8010 (decision-image)
                  rate limit                       registry
```

| Part | Runs on | Managed by |
|---|---|---|
| Ingress `decide.ai.huynle.com` | Traefik on k3s, namespace `decision-api`, middleware `public-chain` (rate limit), wildcard TLS | infra `apps/homelab/decision-api/` |
| Proxy ([`services/proxy`](../services/proxy)) | k3s Deployment `decision-api`, image `ghcr.io/huynle/decision-api-proxy` | CI in this repo + Flux image automation in infra |
| Laya text backend ([`services/laya/serve_local.py`](../services/laya/serve_local.py)) | ml, systemd unit `laya`, port 8000 | infra `hosts/ml/decision-api/` |
| SigLIP2 image backend ([`services/image`](../services/image)) | ml, systemd unit `decision-image`, port 8010 | infra `hosts/ml/decision-api/` |

- `/healthz` and `/v1/health` need no key; every other route needs `Authorization: Bearer <key>`.
- LAN clients resolve `decide.ai.huynle.com` through Pi-hole to `10.1.10.30`. The router does no
  hairpin NAT, so a LAN client that resolves the name to the WAN IP will not connect.
- The backends have no auth. ufw on ml admits only the k3s node (`10.1.10.30`, which pod egress is
  SNATed to) on 8000/8010, and the NetworkPolicy lets only the proxy pod reach ml on those ports.
- Both backends are pinned to the Tesla P40 by GPU UUID (see [GPU switching](#gpu-switching)).

SSH to ml from the laptop (the `ml` alias does not resolve, so use the IP):

```bash
ssh -i ~/.ssh/id_vms huy@10.1.10.10
```

## Deploy proxy changes

Promotion is **manual** since the SQLite key store (ADR 0033 in `huynle/infra`): the image repository in
`apps/homelab/decision-api/image-source.yaml` is suspended and there is no image policy.

1. Merge a PR to `main` that touches `services/proxy/**`.
2. [`.github/workflows/proxy-image.yml`](../.github/workflows/proxy-image.yml) tests the proxy, smoke-tests the
   container and pushes `ghcr.io/huynle/decision-api-proxy:git-<UTC timestamp>-<12-char sha>`. Copy the tag and
   `sha256:` digest from the run.
3. In an infra worktree: `scripts/promote-manual-image.py decision-api <tag> <digest>`, `scripts/validate-infra.sh`,
   open and merge the PR. Check for new `services/proxy/migrations/` files and a recent backup first; the full
   procedure is `docs/runbooks/manual-image-promotion.md#promote-decision-api` in infra.
4. Check: `kubectl -n decision-api rollout status deploy/decision-api`, `curl -s
   https://decide.ai.huynle.com/v1/health` and the [smoke tests](#health-checks-and-smoke-tests). The release uses
   `Recreate`, so each promotion is a brief outage.

To see what runs:
`kubectl -n decision-api get deploy decision-api -o jsonpath='{.spec.template.spec.containers[0].image}'`.

## Deploy backend changes

Backends run from a pinned commit of this repo, checked out in `~/homelab-ai` on ml. From the infra
repo root on the laptop:

```bash
hosts/ml/decision-api/deploy.sh --dry-run <full 40-char homelab-ai sha>   # review
hosts/ml/decision-api/deploy.sh <full 40-char homelab-ai sha>
```

`deploy.sh` checks out the sha (detached HEAD), installs `laya.service` / `decision-image.service` and the
env files in `/etc/decision-api/`, restarts only units whose code, unit, or env changed, and waits for
`/health` on both ports. Update the pinned sha in the script header (and the infra changelog) in the same
infra PR. Dependencies are not installed by `deploy.sh`: if `services/image/requirements.txt` changed,
install it into the image venv on ml first, and check `df -h /` before anything large.

ml runs **Python 3.10**. CI tests the proxy and image server on 3.10 and 3.12; avoid 3.11+ APIs (for
example `tomllib`, `typing.Self`, `except*`, `datetime.UTC`, `binascii.a2b_base64(strict_mode=...)`) in
backend code. A unit that exits right after start usually shows an `AttributeError` or `ImportError` in
`journalctl -u <unit>`.

## Add a model

1. Start a backend on ml (or another host) and check `curl -s 127.0.0.1:<port>/health` there. For a
   permanent backend, add a unit and env file under infra `hosts/ml/decision-api/` and deploy it.
2. If it listens on a new port, open it for `10.1.10.30` only (extend `firewall.sh`) and allow it in the
   NetworkPolicy egress in `release.yaml`.
3. Add an entry to infra `apps/homelab/decision-api/registry.yaml` (schema:
   [proxy README](../services/proxy/README.md#backend-registry)). An existing adapter needs no code change;
   a new kind of backend needs a new adapter in `services/proxy/adapters.py` first.
4. Merge the infra PR, then restart the proxy (the ConfigMap has no hash suffix, so it does not roll the
   pod by itself): `kubectl -n decision-api rollout restart deploy/decision-api`, run through the infra
   GitOps workflow.
5. `GET /v1/models` lists the new id and `GET /v1/health` reports it under `models`.

## Paid OpenRouter models

`POST /api/alpha/decisions` can forward to OpenRouter models (registry entries with `adapter: openrouter`,
for example `jev` = `typesafe/jev-1.14`). They cost OpenRouter credits, so they are off for every key until
the owner turns them on.

### OpenRouter key

1. Create a key at https://openrouter.ai/settings/keys with a **credit limit** (the proxy has no budget of
   its own). Never paste it into Brain, chat or a shared terminal.
2. Store it as a SOPS-encrypted Secret in infra `apps/homelab/decision-api/` (one line, the key only), mount
   it read-only into the pod (`defaultMode: 0400`, readable by uid 10001), and set
   `DECISION_API_OPENROUTER_KEY_FILE` to the mounted path. NetworkPolicy egress must allow `openrouter.ai`
   on 443.
3. Restart the proxy. An unreadable or empty key file stops the proxy at startup. Without the variable the
   proxy starts with a warning and OpenRouter models answer `503 backend_unavailable`;
   `GET /v1/health` shows `"upstreams": {"jev": {"status": "not_probed", "configured": false}}`.
4. Rotate: replace the Secret value, restart, then delete the old key in OpenRouter.

Optional: `DECISION_API_OPENROUTER_REFERER` (default `https://decide.ai.huynle.com`) is the `HTTP-Referer`
OpenRouter uses for app attribution.

### Registry models and aliases

Add the model and the aliases to infra `apps/homelab/decision-api/registry.yaml` (copy the `jev` entry and
`aliases` from [`registry.example.yaml`](../services/proxy/registry.example.yaml)): `capabilities` must
list what the model accepts (`text`, `image`, `box`, `audio`, `video`). `auto: {text: laya-english,
image: jev}` keeps text on the free local Laya and sends images, assets and box questions to `jev`. Merge,
restart the proxy, and check `GET /v1/models` (`paid: true`, `aliases`). OpenRouter models are reachable
only through `POST /api/alpha/decisions`, never the `/v1` routes.

### Enable paid access for a key

SQLite key store (admin page): as the owner, open `/admin/keys` and press **Allow paid models** on the key
(or `PATCH /admin/api/keys/{id}` with `{"allow_paid": true}`). It applies on the key's next request and is
audited as `key.paid_access_changed`; **Deny paid models** turns it off the same way. Members see the value
but cannot change it. Keys without paid access get `403 forbidden` for paid models and nothing is sent to
OpenRouter. In file mode (`KEYSTORE=file`) paid access is all or nothing:
`DECISION_API_FILE_KEYS_ALLOW_PAID=1` (meant for dev and tests).

### Cost

OpenRouter's `usage.cost` (USD credits) is returned to the caller, summed per key and day in the admin
usage report (`cost_usd`, by day, key, model or route), and counted in the metric
`decision_api_upstream_cost_usd_total{model}`. Check the OpenRouter dashboard for the authoritative bill.

## API key management

Keys are lines in the SOPS-encrypted Secret `decision-api-keys` in infra
`apps/homelab/decision-api/decision-api-keys.sops.yaml`: one key per line, `#` starts a comment, keys
must not contain `#`. The proxy reads them once at startup. A self-service admin page is planned but
not built yet.

**Never commit plaintext keys, paste them into Brain or chat, or echo them in a shared terminal.**

### Create a key

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'   # copy into a password manager
sops apps/homelab/decision-api/decision-api-keys.sops.yaml       # add "# <caller>, <date>" and the key
```

Open a PR in infra, merge it, wait for Flux to apply the Secret, then restart the proxy
(`kubectl -n decision-api rollout restart deploy/decision-api`). Hand the key to the caller over a
private channel.

### Rotate a key

1. Add the new key next to the old one, merge, restart. Both keys now work.
2. Move the caller to the new key and confirm its requests succeed (the access log's `key_id` is the
   first 8 hex characters of the key's sha256).
3. Delete the old line, merge, restart.

### Revoke a key

Delete its line, merge, restart.

## Health checks and smoke tests

Public (no key):

```bash
curl -s https://decide.ai.huynle.com/healthz
curl -s https://decide.ai.huynle.com/v1/health
# {"status":"ok","backends":{"text":{"status":"ok",...},"image":{"status":"ok",...}},"models":{...}}
```

Authenticated, from a checkout of this repo, with the key in `DECISION_API_KEY`:

```bash
API=https://decide.ai.huynle.com
curl -s -H "Authorization: Bearer $DECISION_API_KEY" $API/v1/models
curl -s -H "Authorization: Bearer $DECISION_API_KEY" -H 'content-type: application/json' \
  -d @api/fixtures/laya-predict.request.json $API/v1/decide/text
curl -s -o /dev/null -w '%{http_code}\n' -X POST $API/v1/decide/text   # expect 401
# OpenAI Decisions format; answers[0].type is "predicate"
curl -s -H "Authorization: Bearer $DECISION_API_KEY" -H 'content-type: application/json' \
  -d '{"model":"gpt-6-luna","input":"Please refund it or I cancel.","questions":[{"type":"predicate","name":"churn","instructions":"Does the user threaten to cancel?"}]}' \
  $API/v1/decisions
```

The image example is in the [README](../README.md#image-classification).

On ml:

```bash
systemctl status laya decision-image
journalctl -u laya -u decision-image --since '1 hour ago'
curl -s 127.0.0.1:8000/health    # Laya
curl -s 127.0.0.1:8010/health    # image (model load takes ~10 s after a restart)
nvidia-smi
```

Proxy, on k3s:

```bash
kubectl -n decision-api get pods
kubectl -n decision-api logs deploy/decision-api   # one JSON access line per request; no bodies or keys
```

Measured through the public route from the LAN (2026-11-02):

| Route | Latency |
|---|---|
| `POST /v1/decide/text` | ~0.16–0.26 s |
| `POST /v1/classify/image` (dog photo) | ~0.35 s |

## GPU switching

Both units pin a GPU in their env file with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=<GPU UUID>` (the Tesla P40 today). The GTX 1070 is free. To move a backend, set
its `CUDA_VISIBLE_DEVICES` in infra `hosts/ml/decision-api/<unit>.env` to the 1070's UUID (noted in the
file; `nvidia-smi -L` lists them), merge, and re-run `deploy.sh` with the current pin; it restarts the
unit because the env file changed.

## Troubleshooting

### `401 unauthorized`

- The header must be exactly `Authorization: Bearer <key>`.
- The key must be in the Secret **and** the pod must have been restarted after the Secret changed.
- Watch for stray whitespace or a trailing newline from copying the key.

### `413 payload_too_large`

- Text: `state` is over 32 KiB as UTF-8 JSON, or the body is over 64 KiB.
- Image: over 8 MiB after base64 decoding (or the body over 12 MiB). Downscale or re-encode as JPEG.
- Too many questions (over 32) or labels (over 64) return `400 invalid_request`, not 413.

### `503 backend_unavailable`

The proxy could not connect to a backend; `/v1/health` shows which one.

1. On ml: `systemctl status laya decision-image` and `curl 127.0.0.1:8000/health` / `:8010/health`.
   If a unit is failed, read `journalctl -u <unit>`. A CUDA out-of-memory error means something else is
   using the GPU (`nvidia-smi`).
2. Firewall: `sudo ufw status` on ml must allow `10.1.10.30` on 8000,8010 (re-apply with infra
   `hosts/ml/decision-api/firewall.sh`).
3. NetworkPolicy: the pod's egress to `10.1.10.10/32` on 8000/8010 must still be allowed in
   `release.yaml`.

`504 backend_timeout` means the backend accepted the request but took longer than the entry's
`timeout_s` (10 s). `502 backend_error` means it answered with an error; check its journal.

For an OpenRouter model (`/api/alpha/decisions`), 503 means no OpenRouter key is configured (see
`upstreams` in `/v1/health`) or OpenRouter itself is unavailable or unreachable (egress to
`openrouter.ai:443`).

### OpenRouter errors on `/api/alpha/decisions`

- `403 forbidden`: the key has no paid access; see [Enable paid access for a key](#enable-paid-access-for-a-key).
- `502` with "upstream rejected the request (credentials or credits)": OpenRouter refused the proxy's key
  (401/403) or the account is out of credits (402). Check the key and credits in OpenRouter.
- `429 rate_limited`: OpenRouter rate-limited the request; back off.
- `400` with "upstream rejected the request: ...": OpenRouter's own validation message (for example a box
  question to a model that cannot answer it).

### Cannot connect at all

- From the LAN, `dig +short decide.ai.huynle.com` must return `10.1.10.30` (Pi-hole). If it returns the
  WAN IP, the connection fails because there is no hairpin NAT; fix the Pi-hole record or the client's
  DNS.
- `429`: the Traefik `public-chain` rate limit; back off.
- Check the pod and ingress: `kubectl -n decision-api get pods,ingress`.

## Rollback

- **Proxy, registry, keys, ingress:** revert the change in infra (PR, merge) and restart the pod if the
  registry or keys changed. A reverted image tag is re-promoted by Flux image automation to the newest
  image, so for a bad proxy build either revert the commit in this repo (CI builds a fixed image) or
  pause image automation for it first (see infra `docs/image-automation.md`) and then pin the previous tag.
- **Backends:** re-run `hosts/ml/decision-api/deploy.sh <previous sha>` and restore the previous pin in
  the script header.

## Known caveats

- **`pad_to_square` is off by default.** Padding keeps the aspect ratio, but it makes scores for tall
  images such as receipts less decisive. On the left Shell receipt crop, the `fuel receipt` label scored
  0.993 unpadded and 0.432 padded; the dog image was unaffected. Send `"pad_to_square": true` only for
  images where squashing distorts the subject.
- **Laya confidence is uncalibrated.** The English Laya checkpoint's `confidence` values are not
  calibrated probabilities. Treat `min_confidence` thresholds as tuning knobs and check them against real
  data.
- **Python 3.10 on ml.** See [Deploy backend changes](#deploy-backend-changes).
- **Disk on ml is tight** (it was about 90% full). Check `df -h /` before downloading models or
  installing large packages.

## Retired

Until 2026-11-02 everything ran in tmux sessions on ml: the proxy on `127.0.0.1:8080` with keys in
`~/decision-api/keys`, and clients used an SSH tunnel (`ssh -N -L 8080:...`). That proxy, its key file,
and the tunnel path are retired; do not use them. `~/decision-api` on ml is kept as rollback material
until 2026-11-16 and can be deleted after that.
