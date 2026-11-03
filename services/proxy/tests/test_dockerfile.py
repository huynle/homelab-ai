"""The image copies modules by name; a module missing from the Dockerfile only fails at container start."""

from __future__ import annotations

import re
from pathlib import Path

PROXY_DIR = Path(__file__).resolve().parents[1]


def test_dockerfile_copies_every_runtime_module():
    dockerfile = (PROXY_DIR / "Dockerfile").read_text()
    copied = set(re.findall(r"\b([a-z_]+\.py)\b", dockerfile))
    modules = {p.name for p in PROXY_DIR.glob("*.py")}
    assert modules - copied == set(), "add these modules to the Dockerfile COPY line"
