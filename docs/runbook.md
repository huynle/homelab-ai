# Decision API runbook (ml VM)

Operations guide for the decision API on the `ml` VM (`10.1.10.10`). The contract is
[`api/decision-api.yaml`](../api/decision-api.yaml); see [`api/README.md`](../api/README.md) for the
envelope, error codes, and limits.

SSH from the laptop (the `ml` host alias does not resolve, so use the IP):

```bash
ssh -i ~/.ssh/id_vms huy@10.1.10.10
```

## Services

All three services bind to localhost only and run in tmux sessions. **Nothing is under systemd: a
reboot stops all of them.** After a reboot, start them again as shown below.

| tmux session | Service | Bind | Python / venv | Log |
|---|---|---|---|---|
| `laya-test` | Laya text backend (`~/laya-test/serve_local.py`) | `127.0.0.1:8000` | `~/laya-test/.venv` | `~/laya-test/server.log` |
| `decision-image` | SigLIP2 image backend (`services/image/app.py`) | `127.0.0.1:8010` | `~/image-test/.venv` | `~/decision-api/image.log` |
| `decision-proxy` | Decision API proxy (`services/proxy/app.py`) | `127.0.0.1:8080` | `~/decision-api/.venv-proxy` | `~/decision-api/proxy.log` |

Files on ml:

| Path | Purpose |
|---|---|
| `~/decision-api/services/`, `~/decision-api/api/` | Deployed code (rsynced from this repo) |
| `~/decision-api/DEPLOYED_COMMIT` | Commit hash that is deployed |
| `~/decision-api/RUNNING.md` | Start/stop notes kept on the VM |
| `~/decision-api/keys` | API keys, mode 600. Never print or commit it |
| `~/decision-api/test-data/` | Latency/accuracy test script and results |

The proxy only calls the backends per request, so its start order does not matter for correctness,
but `/v1/health` reports `backend_unavailable` until both backends are up.

## Start, stop, restart

Run these on ml. The commands match `~/decision-api/RUNNING.md` (and, for Laya, the command its
running tmux session was started with).

### Laya text backend (`laya-test`)

The Laya session predates this deployment and is shared with other Laya testing. Only restart it
when it is actually down.

```bash
# start
tmux new-session -d -s laya-test "cd ~/laya-test && LAYA_CUDA_AMP=fp32 HF_HUB_OFFLINE=1 PORT=8000 .venv/bin/python -I serve_local.py > ~/laya-test/server.log 2>&1"
until curl -sf 127.0.0.1:8000/health; do sleep 2; done

# stop
tmux kill-session -t laya-test
```

Note that this start command truncates `server.log` (`>`), so copy the log first if you need it.

### Image backend (`decision-image`)

```bash
# start
tmux new-session -d -s decision-image "cd ~/decision-api/services/image && HOST=127.0.0.1 PORT=8010 HF_HOME=$HOME/image-test/models/hf-home HF_HUB_OFFLINE=1 exec $HOME/image-test/.venv/bin/python app.py >> $HOME/decision-api/image.log 2>&1"
until curl -sf 127.0.0.1:8010/health; do sleep 2; done   # model load ~10 s

# stop (frees ~1 GB GPU memory)
tmux kill-session -t decision-image
```

The server only accepts connections once the model has loaded, so "connection refused" right after
start means it is still loading.

### Proxy (`decision-proxy`)

```bash
# start
tmux new-session -d -s decision-proxy "cd ~/decision-api/services/proxy && HOST=127.0.0.1 PORT=8080 DECISION_API_KEYS_FILE=$HOME/decision-api/keys TEXT_BACKEND_URL=http://127.0.0.1:8000 IMAGE_BACKEND_URL=http://127.0.0.1:8010 exec $HOME/decision-api/.venv-proxy/bin/python app.py >> $HOME/decision-api/proxy.log 2>&1"
curl -s 127.0.0.1:8080/v1/health

# stop
tmux kill-session -t decision-proxy
```

### Restart

A restart is stop then start. For example, to restart the proxy:

```bash
tmux kill-session -t decision-proxy
tmux new-session -d -s decision-proxy "cd ~/decision-api/services/proxy && HOST=127.0.0.1 PORT=8080 DECISION_API_KEYS_FILE=$HOME/decision-api/keys TEXT_BACKEND_URL=http://127.0.0.1:8000 IMAGE_BACKEND_URL=http://127.0.0.1:8010 exec $HOME/decision-api/.venv-proxy/bin/python app.py >> $HOME/decision-api/proxy.log 2>&1"
curl -s 127.0.0.1:8080/v1/health
```

Check which sessions are running with `tmux ls`.

## Deploy or update

Run from the laptop, in the checkout or worktree you want to deploy. Commit first so that
`DEPLOYED_COMMIT` names a real commit.

```bash
# 1. copy code (no caches or local venvs)
rsync -av -e "ssh -i ~/.ssh/id_vms" \
  --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude .venv \
  api services huy@10.1.10.10:~/decision-api/

# 2. record the deployed commit
git rev-parse HEAD | ssh -i ~/.ssh/id_vms huy@10.1.10.10 'cat > ~/decision-api/DEPLOYED_COMMIT'
```

3. If `services/proxy/requirements.txt` or `services/image/requirements.txt` changed, install into
   the matching venv on ml first (`~/decision-api/.venv-proxy/bin/pip install -r
   ~/decision-api/services/proxy/requirements.txt`, or `~/image-test/.venv/bin/pip install -r
   ~/decision-api/services/image/requirements.txt`). The disk is about 90% full, so check
   `df -h /` before installing anything large.
4. On ml, restart the sessions whose code changed (`decision-image` and/or `decision-proxy`, see
   [Restart](#restart)). Laya is not deployed from this repo.
5. Run the health check:

   ```bash
   ssh -i ~/.ssh/id_vms huy@10.1.10.10 'curl -s 127.0.0.1:8080/v1/health'
   ```

   Expect `"status":"ok"` for both backends, then run the [smoke tests](#health-checks-and-smoke-tests).

ml runs **Python 3.10**. Before deploying, make sure the code does not use 3.11+ APIs (see
[Troubleshooting](#python-310-compatibility)).

## API key management

Keys live in `~/decision-api/keys` on ml: one key per line, `#` starts a comment, mode `600`. The
proxy reads the file at startup, so every change needs a proxy restart. The proxy refuses to start
with no keys and logs a warning if the file is readable by group or others.

**Never commit keys, paste them into Brain, or echo them in a shared terminal.** The repo's
`.gitignore` excludes `*.key` and `.env`, but the key file is not in the repo at all; keep it that
way.

### Create a key

On ml:

```bash
umask 077
{ echo "# <caller name>, $(date +%F)"; python3 -c 'import secrets; print(secrets.token_urlsafe(32))'; } >> ~/decision-api/keys
chmod 600 ~/decision-api/keys
```

Then restart the proxy. Hand the key to the caller over a private channel (for example, copy it with
`tail -1 ~/decision-api/keys` into a password manager, not into chat or logs). Keys must not contain
`#`.

### Rotate a key

1. Add the new key line (as in [Create a key](#create-a-key)) and keep the old one.
2. Restart the proxy. Both keys now work.
3. Move the caller to the new key and confirm its requests succeed.
4. Delete the old key's line (and its comment) from `~/decision-api/keys`, check
   `stat -c %a ~/decision-api/keys` still prints `600`, and restart the proxy again.

### Revoke a key

Delete its line and restart the proxy.

## Health checks and smoke tests

`GET /v1/health` is public (no key needed):

```bash
curl -s 127.0.0.1:8080/v1/health
# {"status":"ok","backends":{"text":{"status":"ok","latency_ms":22.7},"image":{"status":"ok","latency_ms":21.4}}}
```

Backends directly: `curl -s 127.0.0.1:8000/health` (Laya), `curl -s 127.0.0.1:8010/health` (image).

Authenticated smoke tests, on ml. The key is read into a variable and never echoed:

```bash
KEY=$(head -1 ~/decision-api/keys)   # line 1 is a key; new keys are appended below it
cd ~/decision-api

# models
curl -s -H "Authorization: Bearer $KEY" 127.0.0.1:8080/v1/models

# text decision (real fixture payload)
curl -s -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d @api/fixtures/laya-predict.request.json 127.0.0.1:8080/v1/decide/text

# image classification (receipt test image)
python3 - <<'EOF' > /tmp/classify.json
import base64, json
b64 = base64.b64encode(open("test-data/receipts-left-shell.jpg", "rb").read()).decode()
print(json.dumps({"image": {"base64": b64, "media_type": "image/jpeg"},
                  "candidate_labels": ["receipt", "dog", "cat", "document"]}))
EOF
curl -s -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d @/tmp/classify.json 127.0.0.1:8080/v1/classify/image
rm /tmp/classify.json

# auth is enforced: expect 401
curl -s -o /dev/null -w '%{http_code}\n' -X POST 127.0.0.1:8080/v1/decide/text
```

`~/decision-api/test-data/decision_tests.py` is the fuller latency/accuracy test; its last results are
in `test-data/results.json`. Measured through the proxy on ml (20 calls after 3 warm-ups):

| Route | Median | p95 |
|---|---|---|
| `POST /v1/decide/text` | 212 ms | 285 ms |
| `POST /v1/classify/image` (dog image) | 243 ms | 292 ms |

### From the laptop

The proxy is localhost-only, so reach it through an SSH tunnel:

```bash
ssh -i ~/.ssh/id_vms -N -L 8080:127.0.0.1:8080 huy@10.1.10.10
# in another terminal
curl -s http://127.0.0.1:8080/v1/health
```

Client examples are in the [README](../README.md#usage).

## Logs

| Service | Log |
|---|---|
| Proxy | `~/decision-api/proxy.log` (one JSON access line per request: method, path, status, latency, key id; no bodies or keys) |
| Image | `~/decision-api/image.log` (label count, image size, latency; no image bytes) |
| Laya | `~/laya-test/server.log` |

```bash
tail -f ~/decision-api/proxy.log
tmux attach -t decision-proxy   # detach with Ctrl-b d; do not Ctrl-c, that stops the service
```

## Troubleshooting

### `503 backend_unavailable`

The proxy could not connect to a backend. `curl -s 127.0.0.1:8080/v1/health` shows which one.

- Check `tmux ls`. If `laya-test` or `decision-image` is missing (for example after a reboot),
  start it.
- If the session exists but the port refuses connections, the image backend may still be loading
  (~10 s), or it crashed: read the end of its log. A CUDA out-of-memory error means something else
  is using the GPU (`nvidia-smi`).
- `504 backend_timeout` means the backend accepted the request but took longer than
  `TEXT_TIMEOUT_S` / `IMAGE_TIMEOUT_S` (10 s default). `502 backend_error` means it answered with
  an error; check its log.

### `401 unauthorized`

- The header must be exactly `Authorization: Bearer <key>`.
- The key must be a line in `~/decision-api/keys` **and** the proxy must have been restarted after
  it was added. Keys are read only at startup.
- Watch for stray whitespace or a trailing newline when the key was copied, and make sure
  `head -1` did not pick up a `#` comment line.

### `413 payload_too_large`

- Text: `state` is over 32 KiB as UTF-8 JSON. Trim the text or send less of it.
- Image: over 8 MiB after base64 decoding. Downscale or re-encode the image (JPEG) before sending.
- Too many questions (over 32) or labels (over 64) return `400 invalid_request`, not 413.

### Tunnel does not work

- `bind: Address already in use`: something on the laptop already uses port 8080. Use another
  local port (`-L 18080:127.0.0.1:8080`, then `http://127.0.0.1:18080`) or stop the other process
  (`lsof -i :8080`).
- `Permission denied (publickey)`: pass `-i ~/.ssh/id_vms` and use `huy@10.1.10.10`, not the `ml`
  alias.
- `channel ... open failed: connect failed: Connection refused` in the tunnel terminal: the tunnel is
  fine, but the proxy is not running on ml. Start `decision-proxy`.
- The connection hangs: check the VM is up (`ping 10.1.10.10`).

### Python 3.10 compatibility

ml runs Python 3.10. Code that passes tests on the laptop (3.13) can still fail on ml. An earlier
proxy build used `binascii.a2b_base64(..., strict_mode=True)` (3.11+) and broke on deploy; commit
`08d77eb` replaced it with `base64.b64decode(..., validate=True)`. Before
deploying, avoid 3.11+ features (for example `tomllib`, `typing.Self`, `ExceptionGroup`/`except*`,
`datetime.UTC`) or check them with `python3.10`. If a service exits right after start, read its log
for an `AttributeError` or `ImportError`.

## Known caveats

- **`pad_to_square` is off by default.** Padding keeps the aspect ratio, but it makes scores for
  tall images such as receipts less decisive. On the left Shell crop fuel receipt, `receipt` scored
  0.993 unpadded and 0.432 padded; the dog image was unaffected. Send `"pad_to_square": true` only
  for images where squashing distorts the subject.
- **Laya confidence is uncalibrated.** The English Laya checkpoint's `confidence` values are not
  calibrated probabilities. Treat `min_confidence` thresholds as tuning knobs and check them against
  real data.
- **Not persistent.** Services run in tmux, not systemd, and stop on reboot.
- **Disk is about 90% full** on ml (`df -h /`). Clean up before downloading models or installing
  large packages.
