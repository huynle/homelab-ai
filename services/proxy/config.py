"""Runtime settings for the decision proxy, read from environment variables."""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("decision_api")

DEFAULT_LAYA_VERSION = "laya-serve-0.4.20"
DEFAULT_IMAGE_MODEL_ID = "siglip2-base-patch16-224"
DEFAULT_IMAGE_MODEL_VERSION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DEFAULT_LAYA_CHECKPOINTS = ("english", "multilingual", "typed-decisions")

TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = field(default=(), repr=False)
    allow_no_auth: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    text_backend_url: str = "http://127.0.0.1:8000"
    image_backend_url: str = "http://127.0.0.1:8010"
    text_timeout_s: float = 10.0
    image_timeout_s: float = 10.0
    health_timeout_s: float = 2.0
    laya_model_version: str = DEFAULT_LAYA_VERSION
    laya_checkpoints: tuple[str, ...] = DEFAULT_LAYA_CHECKPOINTS
    image_model_id: str = DEFAULT_IMAGE_MODEL_ID
    image_model_version: str = DEFAULT_IMAGE_MODEL_VERSION
    registry_file: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        d = cls()

        def get(name: str, default):
            value = env.get(name)
            return default if value is None or value.strip() == "" else value.strip()

        return cls(
            api_keys=load_api_keys(env),
            allow_no_auth=get("DECISION_API_ALLOW_NO_AUTH", "0").lower() in TRUTHY,
            host=get("HOST", d.host),
            port=int(get("PORT", d.port)),
            text_backend_url=get("TEXT_BACKEND_URL", d.text_backend_url).rstrip("/"),
            image_backend_url=get("IMAGE_BACKEND_URL", d.image_backend_url).rstrip("/"),
            text_timeout_s=float(get("TEXT_TIMEOUT_S", d.text_timeout_s)),
            image_timeout_s=float(get("IMAGE_TIMEOUT_S", d.image_timeout_s)),
            health_timeout_s=float(get("HEALTH_TIMEOUT_S", d.health_timeout_s)),
            laya_model_version=get("LAYA_MODEL_VERSION", d.laya_model_version),
            image_model_id=get("IMAGE_MODEL_ID", d.image_model_id),
            image_model_version=get("IMAGE_MODEL_VERSION", d.image_model_version),
            registry_file=get("DECISION_API_REGISTRY_FILE", d.registry_file),
        )


def _parse_key_file(path: Path) -> list[str]:
    keys = []
    for line in path.read_text().splitlines():
        key = line.split("#", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def load_api_keys(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Keys from DECISION_API_KEYS_FILE (one per line, # comments) then DECISION_API_KEYS (comma-separated)."""
    env = os.environ if env is None else env
    keys: list[str] = []

    file_name = (env.get("DECISION_API_KEYS_FILE") or "").strip()
    if file_name:
        path = Path(file_name)
        try:
            mode = path.stat().st_mode
            keys.extend(_parse_key_file(path))
        except OSError as exc:
            raise RuntimeError(f"DECISION_API_KEYS_FILE {file_name!r} is not readable ({exc.strerror})") from None
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            log.warning("DECISION_API_KEYS_FILE %s is accessible by group/others; chmod 600 it", file_name)

    keys.extend(k.strip() for k in (env.get("DECISION_API_KEYS") or "").split(","))

    unique: list[str] = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return tuple(unique)
