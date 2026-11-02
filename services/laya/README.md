# Laya text backend launcher

`serve_local.py` runs the HTTP server that ships with Laya (`examples/server.py` in the Laya
repository) against checkpoints already on disk, so the text backend of the decision API starts
offline and never downloads from the Hugging Face Hub.

## Why it patches `examples/server.py`

`examples/server.py` builds its `Router(...)` in the FastAPI lifespan with no option for a local
models directory; it resolves the default Hub bundle. The launcher imports the module, replaces
its `Router` name with `functools.partial(Router, models={...local dirs...})`, and serves
`server.app` with uvicorn. The rest of `server.py` (routes, validation, `LAYA_*` knobs) is used
unchanged, so the launcher tracks whatever Laya version is checked out in `LAYA_REPO_DIR`.

## Requirements

- A Python environment with `laya[serve]` installed (tested with 0.4.20 on Python 3.10).
- A clone of the Laya repo (for `examples/server.py`).
- The three checkpoints in `LAYA_MODELS_DIR`: `laya/`, `laya-multilingual/`,
  `laya-typed-decisions/`. Model files are not part of this repository.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` only behind a firewall: the server has no auth. |
| `PORT` | `8000` | Bind port. |
| `LAYA_MODELS_DIR` | `~/laya-test/models` | Directory with the three checkpoint folders. |
| `LAYA_REPO_DIR` | `~/laya-test/repo` | Laya repo clone; `examples/server.py` is loaded from here. |

Read by Laya / `server.py` themselves: `LAYA_CUDA_AMP` (the `ml` host uses `fp32`),
`LAYA_MAX_LOADED`, `LAYA_DEFAULT_MODEL`, `HF_HUB_OFFLINE=1` (recommended).

## Run

```bash
LAYA_CUDA_AMP=fp32 HF_HUB_OFFLINE=1 python -I serve_local.py
curl -s 127.0.0.1:8000/health
```

The launcher exits with a clear message if the repo or a checkpoint directory is missing.
Production deployment (systemd unit, env file, firewall) lives in `huynle/infra`
under `hosts/ml/decision-api/`.
