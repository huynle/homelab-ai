"""Backend registry: loading, validation, env fallback and model resolution."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from config import Settings
from errors import ApiError
from registry import RegistryError, load_registry, registry_from_settings

PROXY_DIR = Path(__file__).resolve().parents[1]

VALID = """
defaults: {text: laya-english, image: siglip2-base-patch16-224}
models:
  - id: laya-english
    task: text-decision
    adapter: laya
    backend_url: http://10.1.10.10:8000/
    backend_model: english
    version: laya-serve-0.4.20
    timeout_s: 7
  - id: laya-multilingual
    task: text-decision
    adapter: laya
    backend_url: http://10.1.10.10:8000
    backend_model: multilingual
    version: laya-serve-0.4.20
  - id: siglip2-base-patch16-224
    task: image-zero-shot
    adapter: zero-shot-image
    backend_url: http://10.1.10.10:8010
    version: 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
"""


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_valid_file_loads(tmp_path):
    reg = load_registry(write(tmp_path, VALID))
    assert [m.id for m in reg.models] == ["laya-english", "laya-multilingual", "siglip2-base-patch16-224"]
    english = reg.get("laya-english")
    assert english.backend_url == "http://10.1.10.10:8000"  # trailing slash stripped
    assert english.backend_model == "english"
    assert english.timeout_s == 7.0
    assert reg.get("laya-multilingual").timeout_s == 10.0  # default
    image = reg.get("siglip2-base-patch16-224")
    assert image.backend_model is None
    assert image.backend == "zero-shot-image"  # listing label defaults to the adapter key
    assert reg.default("text-decision").id == "laya-english"
    assert reg.default("image-zero-shot").id == "siglip2-base-patch16-224"
    assert reg.backend_urls() == ["http://10.1.10.10:8000", "http://10.1.10.10:8010"]


def test_example_registry_loads():
    reg = load_registry(PROXY_DIR / "registry.example.yaml")
    assert reg.default("text-decision").backend_url == "http://10.1.10.10:8000"
    assert reg.default("image-zero-shot").backend_url == "http://10.1.10.10:8010"


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda t: t.replace("id: laya-multilingual", "id: laya-english"), "duplicate model id 'laya-english'"),
        (lambda t: t.replace("adapter: zero-shot-image", "adapter: clip-magic"), "unknown adapter 'clip-magic'"),
        (lambda t: t.replace("text: laya-english", "text: laya-nope"), "default text model 'laya-nope'"),
        (lambda t: t.replace("image: siglip2-base-patch16-224", "image: laya-english"), "default image model"),
        (
            lambda t: t.replace(
                "task: image-zero-shot\n    adapter: zero-shot-image",
                "task: text-decision\n    adapter: zero-shot-image",
            ),
            "adapter 'zero-shot-image' serves image-zero-shot",
        ),
        (lambda t: t.replace("backend_url: http://10.1.10.10:8010", "backend_url: ftp://x"), "backend_url"),
        (lambda t: t.replace("timeout_s: 7", "timeout_s: 0"), "timeout_s"),
        (lambda t: t.replace("version: laya-serve-0.4.20\n    timeout_s", "colour: red\n    timeout_s"), "colour"),
        (lambda t: "defaults: {text: laya-english}\n" + t.split("\n", 2)[2], "image"),
        (lambda t: "- just\n- a list\n", "mapping"),
        (lambda t: "models: [\n", "YAML"),
    ],
)
def test_invalid_registry_refuses_with_clear_message(tmp_path, mutate, needle):
    with pytest.raises(RegistryError) as info:
        load_registry(write(tmp_path, mutate(textwrap.dedent(VALID))))
    message = str(info.value)
    assert "registry.yaml" in message
    assert needle in message


def test_missing_registry_file_is_an_error(tmp_path):
    with pytest.raises(RegistryError, match="not readable"):
        load_registry(tmp_path / "nope.yaml")


def test_env_fallback_builds_the_old_default_registry():
    reg = registry_from_settings(
        Settings(text_backend_url="http://laya:8000", image_backend_url="http://img:9000", text_timeout_s=3.0)
    )
    assert [(m.id, m.task, m.adapter, m.backend, m.backend_model) for m in reg.models] == [
        ("laya-english", "text-decision", "laya", "laya", "english"),
        ("laya-multilingual", "text-decision", "laya", "laya", "multilingual"),
        ("laya-typed-decisions", "text-decision", "laya", "laya", "typed-decisions"),
        ("siglip2-base-patch16-224", "image-zero-shot", "zero-shot-image", "siglip2", None),
    ]
    assert {m.backend_url for m in reg.models if m.task == "text-decision"} == {"http://laya:8000"}
    assert reg.get("laya-english").timeout_s == 3.0
    assert reg.get("siglip2-base-patch16-224").backend_url == "http://img:9000"
    assert reg.get("siglip2-base-patch16-224").version == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert reg.default("text-decision").id == "laya-english"
    assert reg.default("image-zero-shot").id == "siglip2-base-patch16-224"


def test_settings_read_registry_file_env():
    s = Settings.from_env({"DECISION_API_KEYS": "k", "DECISION_API_REGISTRY_FILE": " /etc/decision-api/registry.yaml "})
    assert s.registry_file == "/etc/decision-api/registry.yaml"
    assert Settings.from_env({"DECISION_API_KEYS": "k"}).registry_file is None


# ---------------------------------------------------------------- resolution


@pytest.fixture
def reg(tmp_path):
    return load_registry(write(tmp_path, VALID))


def test_absent_model_resolves_to_task_default(reg):
    assert reg.resolve("text-decision", None) == (reg.get("laya-english"), False)
    assert reg.resolve("image-zero-shot", None) == (reg.get("siglip2-base-patch16-224"), False)


def test_model_id_selects_entry(reg):
    assert reg.resolve("text-decision", "laya-multilingual") == (reg.get("laya-multilingual"), True)


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("english", "laya-english"),
        ("multilingual", "laya-multilingual"),
        ("en", "laya-english"),
        ("multi", "laya-multilingual"),
    ],
)
def test_laya_aliases_map_to_registry_ids(reg, alias, expected):
    assert reg.resolve("text-decision", alias) == (reg.get(expected), True)


def test_alias_without_matching_entry_is_unknown(reg):
    with pytest.raises(ApiError) as info:
        reg.resolve("text-decision", "typed-decisions")
    assert info.value.code == "invalid_request"
    assert "laya-english, laya-multilingual" in info.value.message


def test_wrong_task_model_is_invalid_request(reg):
    with pytest.raises(ApiError) as info:
        reg.resolve("text-decision", "siglip2-base-patch16-224")
    assert info.value.code == "invalid_request"
    assert "image-zero-shot" in info.value.message
    assert "laya-english" in info.value.message


def test_unknown_model_lists_valid_ids(reg):
    with pytest.raises(ApiError) as info:
        reg.resolve("image-zero-shot", "klingon")
    assert info.value.code == "invalid_request"
    assert "'klingon'" in info.value.message
    assert "siglip2-base-patch16-224" in info.value.message
