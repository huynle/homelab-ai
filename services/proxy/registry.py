"""Backend registry: which models the proxy serves, where their backends live, and how to talk to them.

Loaded once at startup from ``DECISION_API_REGISTRY_FILE`` (YAML, see ``registry.example.yaml``). When that
variable is unset, :func:`registry_from_settings` builds the registry the proxy always had from the
``TEXT_BACKEND_URL`` / ``IMAGE_BACKEND_URL`` environment, so existing deployments keep working.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from adapters import ADAPTERS
from errors import ApiError

log = logging.getLogger("decision_api")

TEXT_TASK = "text-decision"
IMAGE_TASK = "image-zero-shot"
DECISIONS_TASK = "decisions"  # OpenRouter-native; served only by POST /api/alpha/decisions
Task = Literal["text-decision", "image-zero-shot", "decisions"]
Capability = Literal["text", "image", "box", "audio", "video"]
DEFAULT_TIMEOUT_S = 10.0

# Laya's own checkpoint aliases (api/fixtures/laya-models.json). A text request may still pin a checkpoint
# the way it did before the registry existed (`english`, `en`, `multi`, ...); it maps to `laya-<checkpoint>`.
LAYA_ALIASES = {
    "decisions": "typed-decisions",
    "default": "english",
    "en": "english",
    "laya": "english",
    "ml": "multilingual",
    "multi": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
}

ModelId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,128}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class RegistryError(RuntimeError):
    """The registry file is missing or invalid; the proxy refuses to start."""


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: ModelId
    task: Task
    adapter: str
    backend_url: str
    backend_model: NonEmpty | None = None
    backend: NonEmpty | None = None  # label shown in /v1/models; defaults to the adapter key
    version: NonEmpty
    timeout_s: Annotated[float, Field(gt=0, le=600, strict=False)] = DEFAULT_TIMEOUT_S
    # What the model accepts. Fixed by the adapter for local backends; required for openrouter entries.
    capabilities: Annotated[list[Capability], Field(min_length=1)] | None = None

    @property
    def paid(self) -> bool:
        """Calls cost money upstream (adapter openrouter); only keys with paid access may use the model."""
        return ADAPTERS[self.adapter].paid

    @property
    def probed(self) -> bool:
        """GET /v1/health probes the backend. OpenRouter has no health endpoint and is never probed."""
        return not self.paid

    @field_validator("backend_url")
    @classmethod
    def _url(cls, url: str) -> str:
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
            raise ValueError("must be an http(s) URL with a host and no query string, e.g. http://10.1.10.10:8000")
        return url.strip().rstrip("/")

    @model_validator(mode="after")
    def _adapter(self) -> ModelEntry:
        adapter = ADAPTERS.get(self.adapter)
        if adapter is None:
            raise ValueError(f"unknown adapter {self.adapter!r} (known: {', '.join(sorted(ADAPTERS))})")
        if adapter.task != self.task:
            raise ValueError(f"adapter {self.adapter!r} serves {adapter.task}, not {self.task}")
        if adapter.capabilities is not None:
            if "capabilities" in self.model_fields_set:
                raise ValueError(
                    f"capabilities are fixed for adapter {self.adapter!r} ({', '.join(adapter.capabilities)}); "
                    "remove the field"
                )
            # bypass fields_set: the value is derived, and the entry is validated again inside Registry
            self.__dict__["capabilities"] = list(adapter.capabilities)
        elif self.capabilities is None:
            raise ValueError(f"adapter {self.adapter!r} needs capabilities (a list of: text, image, box, audio, video)")
        else:
            self.capabilities = list(dict.fromkeys(self.capabilities))
        if self.adapter == "openrouter":
            if self.backend_model is None:
                raise ValueError("adapter 'openrouter' needs backend_model (the OpenRouter model id)")
            if urlsplit(self.backend_url).scheme != "https":
                raise ValueError("adapter 'openrouter' needs an https backend_url, e.g. https://openrouter.ai/api")
        if self.backend is None:
            self.backend = self.adapter
        return self


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    image: str


DEFAULT_TASKS = {"text": TEXT_TASK, "image": IMAGE_TASK}


class AliasTargets(BaseModel):
    """``{text: <id>, image: <id>}``: the image target serves requests with media or box questions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: ModelId
    image: ModelId


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    defaults: Defaults
    models: Annotated[list[ModelEntry], Field(min_length=1)]
    # Names for POST /api/alpha/decisions only: `name: <model id>` or `name: {text: <id>, image: <id>}`.
    aliases: dict[ModelId, ModelId | AliasTargets] = Field(default_factory=dict)
    _by_id: dict[str, ModelEntry] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Registry:
        by_id: dict[str, ModelEntry] = {}
        for entry in self.models:
            if entry.id in by_id:
                raise ValueError(f"duplicate model id {entry.id!r}")
            by_id[entry.id] = entry
        for key, task in DEFAULT_TASKS.items():
            wanted = getattr(self.defaults, key)
            entry = by_id.get(wanted)
            if entry is None:
                raise ValueError(f"default {key} model {wanted!r} is not in models")
            if entry.task != task:
                raise ValueError(f"default {key} model {wanted!r} serves {entry.task}, not {task}")
        for name, target in self.aliases.items():
            if name in by_id:
                raise ValueError(f"alias {name!r} has the same name as a model id")
            targets = (target,) if isinstance(target, str) else (target.text, target.image)
            for t in targets:
                if t in self.aliases:
                    raise ValueError(f"alias {name!r} points at alias {t!r}; aliases must name model ids")
                if t not in by_id:
                    raise ValueError(f"alias {name!r} points at {t!r}, which is not in models")
        self._by_id = by_id
        return self

    # ------------------------------------------------------------ lookups

    def find(self, model_id: str) -> ModelEntry | None:
        return self._by_id.get(model_id)

    def get(self, model_id: str) -> ModelEntry:
        return self._by_id[model_id]

    def default(self, task: str) -> ModelEntry:
        key = next(k for k, t in DEFAULT_TASKS.items() if t == task)
        return self._by_id[getattr(self.defaults, key)]

    def for_task(self, task: str) -> list[ModelEntry]:
        return [m for m in self.models if m.task == task]

    def probed(self) -> list[ModelEntry]:
        """Models whose backend GET /v1/health probes (not openrouter)."""
        return [m for m in self.models if m.probed]

    def backend_urls(self) -> list[str]:
        """Distinct probed backend URLs in registry order."""
        return list(dict.fromkeys(m.backend_url for m in self.probed()))

    def alias_list(self) -> list[dict]:
        """Aliases as ``{id, text, image}`` (a single-target alias has the same id twice)."""
        out = []
        for name, target in self.aliases.items():
            text, image = (target, target) if isinstance(target, str) else (target.text, target.image)
            out.append({"id": name, "text": text, "image": image})
        return out

    def resolve_decision(self, name: str, *, media: bool) -> ModelEntry:
        """The model for POST /api/alpha/decisions: a model id, an alias, or a Laya checkpoint name.

        ``media``: the request has media parts, assets or box questions, so an alias picks its image target.
        """
        target = self.aliases.get(name)
        if target is not None:
            if not isinstance(target, str):
                target = target.image if media else target.text
            return self._by_id[target]
        entry = self.find(name)
        if entry is None:
            candidate = self.find(f"laya-{LAYA_ALIASES.get(name, name)}")
            if candidate is not None and candidate.adapter == "laya":
                entry = candidate
        if entry is None:
            shown = name if len(name) <= 80 else name[:80] + "..."
            valid = ", ".join([m.id for m in self.models] + list(self.aliases))
            raise ApiError("invalid_request", f"unknown model {shown!r}; valid models and aliases: {valid}", param="model")
        return entry

    def resolve(self, task: str, name: str | None) -> tuple[ModelEntry, bool]:
        """Pick the entry for a request: ``(entry, explicit)``. Unknown or wrong-task names raise 400."""
        if name is None:
            return self.default(task), False
        entry = self.find(name)
        if entry is None and task == TEXT_TASK:
            candidate = self.find(f"laya-{LAYA_ALIASES.get(name, name)}")
            if candidate is not None and candidate.adapter == "laya":
                entry = candidate
        valid = ", ".join(m.id for m in self.for_task(task))
        shown = name if len(name) <= 80 else name[:80] + "..."
        if entry is None:
            raise ApiError("invalid_request", f"unknown model {shown!r}; valid {task} models: {valid}")
        if entry.task != task:
            raise ApiError(
                "invalid_request", f"model {shown!r} serves {entry.task}, not {task}; valid {task} models: {valid}"
            )
        return entry, True


# ---------------------------------------------------------------- construction


def _errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_input=False, include_url=False)[:10]:
        loc = ".".join(str(p) for p in err["loc"]) or "registry"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def load_registry(path: str | Path) -> Registry:
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise RegistryError(f"registry file {str(path)!r} is not readable ({exc.strerror})") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegistryError(f"invalid registry {path}: not valid YAML ({exc})") from None
    if not isinstance(data, dict):
        raise RegistryError(f"invalid registry {path}: the top level must be a mapping with defaults and models")
    try:
        registry = Registry.model_validate(data)
    except ValidationError as exc:
        raise RegistryError(f"invalid registry {path}: {_errors(exc)}") from None
    log.info("loaded registry %s: %s", path, ", ".join(m.id for m in registry.models))
    return registry


def registry_from_settings(settings) -> Registry:
    """The pre-registry setup: Laya checkpoints on TEXT_BACKEND_URL, one SigLIP2 model on IMAGE_BACKEND_URL."""
    text = [
        ModelEntry(
            id=f"laya-{checkpoint}",
            task=TEXT_TASK,
            adapter="laya",
            backend_url=settings.text_backend_url,
            backend_model=checkpoint,
            version=settings.laya_model_version,
            timeout_s=settings.text_timeout_s,
        )
        for checkpoint in settings.laya_checkpoints
    ]
    image = ModelEntry(
        id=settings.image_model_id,
        task=IMAGE_TASK,
        adapter="zero-shot-image",
        backend="siglip2",
        backend_url=settings.image_backend_url,
        version=settings.image_model_version,
        timeout_s=settings.image_timeout_s,
    )
    return Registry(defaults=Defaults(text=text[0].id, image=image.id), models=[*text, image])
