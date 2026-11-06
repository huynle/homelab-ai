"""Registry: the openrouter adapter, capabilities, the paid flag and aliases (POST /api/alpha/decisions)."""

from __future__ import annotations

import textwrap

import pytest

from errors import ApiError
from registry import RegistryError, load_registry

BASE = """
defaults: {text: laya-english, image: siglip2-base-patch16-224}
models:
  - id: laya-english
    task: text-decision
    adapter: laya
    backend_url: http://10.1.10.10:8000
    backend_model: english
    version: laya-serve-0.4.20
  - id: siglip2-base-patch16-224
    task: image-zero-shot
    adapter: zero-shot-image
    backend_url: http://10.1.10.10:8010
    version: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
  - id: jev
    task: decisions
    adapter: openrouter
    backend_url: https://openrouter.ai/api
    backend_model: typesafe/jev-1.14
    capabilities: [text, image, box]
    version: jev-1.14
    timeout_s: 60
aliases:
  auto: {text: laya-english, image: jev}
  fast: laya-english
"""


def load(tmp_path, text: str = BASE):
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(text))
    return load_registry(path)


def test_openrouter_entry_loads(tmp_path):
    reg = load(tmp_path)
    jev = reg.get("jev")
    assert jev.task == "decisions"
    assert jev.backend_model == "typesafe/jev-1.14"
    assert jev.capabilities == ["text", "image", "box"]
    assert jev.paid is True
    assert jev.timeout_s == 60.0


def test_local_capabilities_are_fixed_by_adapter(tmp_path):
    reg = load(tmp_path)
    assert reg.get("laya-english").capabilities == ["text"]
    assert reg.get("laya-english").paid is False
    assert reg.get("siglip2-base-patch16-224").capabilities == ["image"]
    assert reg.get("siglip2-base-patch16-224").paid is False


def test_openrouter_is_not_probed(tmp_path):
    reg = load(tmp_path)
    assert reg.backend_urls() == ["http://10.1.10.10:8000", "http://10.1.10.10:8010"]
    assert [m.id for m in reg.probed()] == ["laya-english", "siglip2-base-patch16-224"]


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda t: t.replace("    capabilities: [text, image, box]\n", ""), "capabilities"),
        (lambda t: t.replace("    backend_model: typesafe/jev-1.14\n", ""), "backend_model"),
        (lambda t: t.replace("https://openrouter.ai/api", "http://openrouter.ai/api"), "https"),
        (lambda t: t.replace("[text, image, box]", "[text, smell]"), "capabilities"),
        (lambda t: t.replace("task: decisions", "task: text-decision"), "adapter 'openrouter' serves decisions"),
        (
            lambda t: t.replace("backend_model: english\n", "backend_model: english\n    capabilities: [text]\n"),
            "capabilities are fixed",
        ),
        (lambda t: t.replace("image: jev}", "image: nope}"), "alias 'auto'"),
        (lambda t: t.replace("fast: laya-english", "fast: auto"), "alias 'fast'"),
        (lambda t: t.replace("fast: laya-english", "jev: laya-english"), "alias 'jev'"),
        (lambda t: t.replace("fast: laya-english", "fast: {text: laya-english}"), "fast"),
        (lambda t: t.replace("fast: laya-english", "Bad Name!: laya-english"), "alias"),
    ],
)
def test_invalid_openrouter_or_alias_config(tmp_path, mutate, needle):
    with pytest.raises(RegistryError) as info:
        load(tmp_path, mutate(textwrap.dedent(BASE)))
    assert needle in str(info.value)


def test_openrouter_cannot_be_a_default(tmp_path):
    text = textwrap.dedent(BASE).replace("text: laya-english, image", "text: jev, image")
    with pytest.raises(RegistryError, match="default text model 'jev'"):
        load(tmp_path, text)


# ---------------------------------------------------------------- resolution for /api/alpha/decisions


@pytest.mark.parametrize(
    "name,media,expected",
    [
        ("auto", False, "laya-english"),
        ("auto", True, "jev"),
        ("fast", True, "laya-english"),
        ("jev", False, "jev"),
        ("laya-english", False, "laya-english"),
        ("english", False, "laya-english"),
        ("siglip2-base-patch16-224", True, "siglip2-base-patch16-224"),
    ],
)
def test_resolve_decision_model(tmp_path, name, media, expected):
    assert load(tmp_path).resolve_decision(name, media=media).id == expected


def test_unknown_decision_model_lists_models_and_aliases(tmp_path):
    with pytest.raises(ApiError) as info:
        load(tmp_path).resolve_decision("gpt-9", media=False)
    err = info.value
    assert err.code == "invalid_request"
    assert err.param == "model"
    for name in ("laya-english", "siglip2-base-patch16-224", "jev", "auto", "fast"):
        assert name in err.message


def test_alias_listing(tmp_path):
    assert load(tmp_path).alias_list() == [
        {"id": "auto", "text": "laya-english", "image": "jev"},
        {"id": "fast", "text": "laya-english", "image": "laya-english"},
    ]


def test_v1_routes_never_resolve_openrouter_entries_or_aliases(tmp_path):
    reg = load(tmp_path)
    for task in ("text-decision", "image-zero-shot"):
        for name in ("jev", "auto"):
            with pytest.raises(ApiError):
                reg.resolve(task, name)
    assert [m.id for m in reg.for_task("text-decision")] == ["laya-english"]
