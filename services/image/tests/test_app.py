import base64
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import Settings, create_app, pad_to_square
from conftest import FakeEncoder, png_b64

LABELS = ["cat", "dog", "car"]


def classify(client, **overrides):
    body = {"image": {"base64": png_b64()}, "candidate_labels": LABELS}
    body.update(overrides)
    return client.post("/classify", json=body)


def assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


# ---------- happy path ----------


def test_classify_orders_by_score_and_softmax_sums_to_one(client):
    resp = classify(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "siglip2-base-patch16-224"
    assert body["model_version"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] >= 0
    labels = [r["label"] for r in body["results"]]
    assert labels == ["dog", "cat", "car"]  # original labels, not filled templates
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert math.isclose(sum(scores), 1.0, abs_tol=1e-4)


def test_scores_are_softmax_of_logit_scale_times_cosine(client):
    body = classify(client).json()
    # Fake cosines: dog 0.9, cat 0.5, car 0.0; logit_scale 10.
    logits = {"dog": 9.0, "cat": 5.0, "car": 0.0}
    z = sum(math.exp(v) for v in logits.values())
    for r in body["results"]:
        assert math.isclose(r["score"], math.exp(logits[r["label"]]) / z, rel_tol=1e-4)


def test_default_template_is_a_photo_of_a(client, encoder):
    classify(client)
    assert encoder.text_calls == [["a photo of a cat", "a photo of a dog", "a photo of a car"]]


def test_custom_template_is_used(client, encoder):
    resp = classify(client, hypothesis_template="an image of {} here")
    assert resp.status_code == 200
    assert encoder.text_calls[-1][0] == "an image of cat here"


def test_top_k_truncates_after_sorting(client):
    body = classify(client, top_k=2).json()
    assert [r["label"] for r in body["results"]] == ["dog", "cat"]


def test_top_k_larger_than_labels_returns_all(client):
    body = classify(client, top_k=10).json()
    assert len(body["results"]) == 3


def test_transparent_png_is_converted_to_rgb(client, encoder):
    import io

    buf = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(buf, format="PNG")
    resp = client.post(
        "/classify",
        json={"image": {"base64": base64.b64encode(buf.getvalue()).decode()}, "candidate_labels": ["x"]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"][0]["score"] == pytest.approx(1.0)


# ---------- caching ----------


def test_text_embeddings_cached_for_same_template_and_labels(client, encoder):
    classify(client)
    classify(client)
    assert len(encoder.text_calls) == 1


def test_cache_misses_on_different_labels_or_template(client, encoder):
    classify(client)
    classify(client, candidate_labels=["dog", "cat"])
    classify(client, hypothesis_template="a picture of {}")
    assert len(encoder.text_calls) == 3


def test_cache_is_bounded(encoder):
    app = create_app(settings=Settings(text_cache_size=1), encoder=encoder)
    with TestClient(app) as c:
        classify(c, candidate_labels=["a"])
        classify(c, candidate_labels=["b"])
        classify(c, candidate_labels=["a"])
    assert len(encoder.text_calls) == 3


# ---------- pad_to_square ----------


def test_pad_to_square_on_tall_image_produces_square(client, encoder):
    resp = classify(client, image={"base64": png_b64((40, 200))})
    assert resp.status_code == 200
    assert encoder.image_sizes[-1] == (200, 200)


def test_pad_to_square_false_keeps_aspect(client, encoder):
    resp = classify(client, image={"base64": png_b64((40, 200))}, pad_to_square=False)
    assert resp.status_code == 200
    assert encoder.image_sizes[-1] == (40, 200)


def test_pad_to_square_centers_with_white_fill():
    img = Image.new("RGB", (2, 6), (0, 0, 0))
    out = pad_to_square(img)
    assert out.size == (6, 6)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((5, 5)) == (255, 255, 255)
    assert out.getpixel((2, 0)) == (0, 0, 0)
    assert out.getpixel((3, 5)) == (0, 0, 0)


def test_pad_to_square_wide_image_and_square_noop():
    assert pad_to_square(Image.new("RGB", (9, 3))).size == (9, 9)
    sq = Image.new("RGB", (5, 5))
    assert pad_to_square(sq).size == (5, 5)


# ---------- validation -> 400 ----------


@pytest.mark.parametrize("template", ["a photo of a", "a photo of a {", "", "x{}" + "y" * 512])
def test_template_must_contain_placeholder(client, template):
    assert_error(classify(client, hypothesis_template=template), 400, "invalid_request")


def test_template_with_other_braces_is_literal(client, encoder):
    resp = classify(client, hypothesis_template="{x} photo of {}")
    assert resp.status_code == 200, resp.text
    assert encoder.text_calls[-1][0] == "{x} photo of cat"


def test_too_few_labels(client):
    assert_error(classify(client, candidate_labels=[]), 400, "invalid_request")


def test_too_many_labels(client):
    assert_error(classify(client, candidate_labels=[f"l{i}" for i in range(65)]), 400, "invalid_request")


def test_64_labels_is_allowed(client):
    resp = classify(client, candidate_labels=[f"l{i}" for i in range(64)])
    assert resp.status_code == 200


def test_duplicate_labels_rejected(client):
    assert_error(classify(client, candidate_labels=["dog", "dog"]), 400, "invalid_request")


def test_empty_label_rejected(client):
    assert_error(classify(client, candidate_labels=["dog", ""]), 400, "invalid_request")


@pytest.mark.parametrize("top_k", [0, -1, 65])
def test_top_k_out_of_range(client, top_k):
    assert_error(classify(client, top_k=top_k), 400, "invalid_request")


def test_invalid_base64(client):
    assert_error(classify(client, image={"base64": "not base64!!"}), 400, "invalid_request")


def test_non_image_bytes(client):
    data = base64.b64encode(b"hello, this is not an image at all").decode()
    assert_error(classify(client, image={"base64": data}), 400, "invalid_request")


def test_missing_image_field(client):
    resp = client.post("/classify", json={"candidate_labels": LABELS})
    assert_error(resp, 400, "invalid_request")


def test_unknown_field_rejected(client):
    assert_error(classify(client, extra=1), 400, "invalid_request")


def test_bad_media_type_rejected(client):
    assert_error(classify(client, image={"base64": png_b64(), "media_type": "image/gif"}), 400, "invalid_request")


def test_malformed_json(client):
    resp = client.post("/classify", content=b"{nope", headers={"content-type": "application/json"})
    assert_error(resp, 400, "invalid_request")


def test_decompression_bomb_rejected(encoder):
    import io

    buf = io.BytesIO()
    Image.new("1", (3000, 3000)).save(buf, format="PNG")  # tiny file, 9M pixels
    app = create_app(settings=Settings(max_pixels=1_000_000), encoder=encoder)
    with TestClient(app) as c:
        resp = c.post(
            "/classify",
            json={"image": {"base64": base64.b64encode(buf.getvalue()).decode()}, "candidate_labels": ["x"]},
        )
    assert_error(resp, 400, "invalid_request")


# ---------- 413 ----------


def test_oversize_image_returns_413(client):
    data = base64.b64encode(b"\0" * (8 * 1024 * 1024 + 1)).decode()
    resp = classify(client, image={"base64": data})
    assert_error(resp, 413, "payload_too_large")
    assert "8388608" in resp.json()["error"]["message"]


def test_max_bytes_is_configurable(encoder):
    app = create_app(settings=Settings(max_bytes=100), encoder=encoder)
    with TestClient(app) as c:
        resp = c.post("/classify", json={"image": {"base64": png_b64((64, 64), "blue")}, "candidate_labels": ["x"]})
    # A PNG of a flat colour is small but still > 100 bytes.
    assert_error(resp, 413, "payload_too_large")


# ---------- 500 ----------


def test_encoder_failure_returns_500_without_leaking(settings):
    app = create_app(settings=settings, encoder=FakeEncoder(fail=True))
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = classify(c)
    assert_error(resp, 500, "internal_error")
    assert "boom" not in resp.text


# ---------- health ----------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model": "siglip2-base-patch16-224",
        "model_version": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "device": "cpu",
    }


# ---------- settings ----------


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL_ID", "org/some-model")
    monkeypatch.setenv("IMAGE_MODEL_REVISION", "abc")
    monkeypatch.setenv("IMAGE_DEVICE", "cpu")
    monkeypatch.setenv("IMAGE_MAX_BYTES", "123")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9999")
    s = Settings.from_env()
    assert (s.model_id, s.revision, s.device, s.max_bytes, s.host, s.port) == (
        "org/some-model", "abc", "cpu", 123, "0.0.0.0", 9999,
    )
    assert s.model_name == "some-model"


def test_settings_defaults(monkeypatch):
    for k in ["IMAGE_MODEL_ID", "IMAGE_MODEL_REVISION", "IMAGE_DEVICE", "IMAGE_MAX_BYTES", "HOST", "PORT"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.model_id == "google/siglip2-base-patch16-224"
    assert s.revision == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert s.device == "auto"
    assert s.max_bytes == 8 * 1024 * 1024
    assert (s.host, s.port) == ("127.0.0.1", 8010)
    assert s.model_name == "siglip2-base-patch16-224"


def test_importing_app_does_not_import_torch_or_transformers():
    import subprocess
    import sys

    code = "import sys, app; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=__file__.rsplit("/tests/", 1)[0])
