"""Run Laya's example HTTP server (examples/server.py) against local, offline checkpoints.

examples/server.py has no models-dir option: its lifespan builds ``Router(...)`` from the Hub
bundle. This launcher imports that module and replaces its ``Router`` name with a partial that
passes ``models=`` pointing at checkpoints on disk, then serves ``server.app`` with uvicorn.
Nothing else in server.py is changed.

Environment:
  HOST             bind address (default 127.0.0.1)
  PORT             bind port (default 8000)
  LAYA_MODELS_DIR  directory holding laya/, laya-multilingual/, laya-typed-decisions/
                   (default ~/laya-test/models)
  LAYA_REPO_DIR    clone of the Laya repo; examples/server.py is loaded from it
                   (default ~/laya-test/repo)
Other variables (LAYA_CUDA_AMP, LAYA_MAX_LOADED, LAYA_DEFAULT_MODEL, HF_HUB_OFFLINE, ...) are
read by laya / server.py themselves.
"""
import functools
import os
import sys


def _path(name: str, default: str) -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get(name) or default))


HOST = os.environ.get("HOST") or "127.0.0.1"
PORT = int(os.environ.get("PORT") or "8000")
MODELS_DIR = _path("LAYA_MODELS_DIR", "~/laya-test/models")
REPO_DIR = _path("LAYA_REPO_DIR", "~/laya-test/repo")

CHECKPOINTS = {
    "english": "laya",
    "multilingual": "laya-multilingual",
    "typed-decisions": "laya-typed-decisions",
}


def local_models(models_dir: str) -> dict:
    """Map Router checkpoint names to local directories; fail fast if one is missing."""
    models = {name: os.path.join(models_dir, sub) for name, sub in CHECKPOINTS.items()}
    missing = [p for p in models.values() if not os.path.isdir(p)]
    if missing:
        raise SystemExit(f"missing Laya checkpoint directories: {', '.join(missing)}")
    return models


def main() -> None:
    examples = os.path.join(REPO_DIR, "examples")
    if not os.path.isfile(os.path.join(examples, "server.py")):
        raise SystemExit(f"examples/server.py not found under LAYA_REPO_DIR={REPO_DIR}")
    models = local_models(MODELS_DIR)
    sys.path.insert(0, examples)

    import server  # examples/server.py
    import uvicorn
    from laya import Router

    server.Router = functools.partial(Router, models=models)
    uvicorn.run(server.app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
