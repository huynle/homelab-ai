# image: SigLIP2 zero-shot classifier

Internal backend for the decision-api proxy (`POST /v1/classify/image` in
[`api/decision-api.yaml`](../../api/decision-api.yaml)). It serves
`google/siglip2-base-patch16-224` at commit `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, fp32, and
listens on `127.0.0.1:8010` by default. It does no authentication; only the proxy should reach it.

## API

`POST /classify`

```json
{"image": {"base64": "<standard base64>", "media_type": "image/jpeg"},
 "candidate_labels": ["dog", "cat", "receipt"],
 "hypothesis_template": "a photo of a {}",
 "top_k": 2,
 "pad_to_square": true}
```

The fields, defaults, and limits match the contract's `ImageClassifyRequest`: 1 to 64 unique
labels of 1 to 256 characters, a template of at most 512 characters that contains `{}` (every `{}`
is replaced by the label, and other braces are kept as they are), `top_k` from 1 to 64, and at
most 8 MiB of image data after decoding.

```json
{"model": "siglip2-base-patch16-224",
 "model_version": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
 "latency_ms": 64.5,
 "results": [{"label": "dog", "score": 0.97}, {"label": "cat", "score": 0.02}]}
```

`results` is sorted by descending `score` and uses the original labels. The scores are a softmax
over the candidate labels of `logit_scale * cosine(image, text)`, the same as `prob` in
`api/fixtures/siglip-*.json`. They sum to about 1 before `top_k` truncation. The proxy can pass
this body through and add `meta`.

`GET /health` returns `{"status": "ok", "model": ..., "model_version": ..., "device": "cuda"}`.
The server only accepts connections after the model has loaded, so a refused connection means it
is not ready yet.

Errors use the contract's shape, `{"error": {"code", "message"}}`:

| status | code | when |
|---|---|---|
| 400 | `invalid_request` | malformed JSON, schema violations (label count, duplicate or empty labels, template without `{}`, `top_k` out of range, unknown fields), invalid base64, bytes that are not an image, images over `IMAGE_MAX_PIXELS` |
| 413 | `payload_too_large` | decoded image larger than `IMAGE_MAX_BYTES` |
| 500 | `internal_error` | inference failure. The message is generic, and only the exception type is logged |

## Behaviour

- **Padding.** With `pad_to_square` (the default), the image is centred on a white square before
  the processor runs. The processor resizes straight to 224x224 and ignores the aspect ratio, so
  without padding tall receipts get squashed. Transparent images are composited onto white.
- **Text cache.** Text embeddings are cached in an LRU keyed by `(template, labels)` with 64
  entries by default. Repeating a label set costs one image forward pass.
- **Inference.** Forward passes are serialized with a lock. FastAPI runs the sync handler in its
  threadpool.
- **Logging.** Logs show the label count, image size, and latency. Image bytes and request bodies
  are never logged.

## Configuration (environment)

| var | default |
|---|---|
| `IMAGE_MODEL_ID` | `google/siglip2-base-patch16-224` |
| `IMAGE_MODEL_REVISION` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` (also returned as `model_version`) |
| `IMAGE_DEVICE` | `auto` (`cuda` if available, else `cpu`) |
| `IMAGE_MAX_BYTES` | `8388608` |
| `IMAGE_MAX_PIXELS` | `64000000` (decompression-bomb guard) |
| `IMAGE_TEXT_CACHE_SIZE` | `64` |
| `HOST` / `PORT` | `127.0.0.1` / `8010` |
| `LOG_LEVEL` | `INFO` |
| `HF_HOME`, `HF_HUB_OFFLINE` | standard Hugging Face cache settings |

## Run on ml

The weights are already in `~/image-test/models/hf-home`. Run fully offline:

```bash
cd services/image
python -m venv .venv && . .venv/bin/activate     # or reuse the env that has torch 2.14.1 + transformers 5.20.2
pip install -r requirements.txt
export HF_HOME=~/image-test/models/hf-home HF_HUB_OFFLINE=1
python app.py                                     # serves on 127.0.0.1:8010
curl -s localhost:8010/health
python - <<'EOF'
import base64, json, urllib.request
img = base64.b64encode(open("/home/huy/image-test/data/receipts-three.jpg", "rb").read()).decode()
body = json.dumps({"image": {"base64": img},
                   "candidate_labels": ["receipt", "fuel receipt", "dog", "landscape"]}).encode()
req = urllib.request.Request("http://127.0.0.1:8010/classify", body, {"content-type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
EOF
```

## Tests

The tests use a deterministic fake encoder (`tests/conftest.py`). They need no GPU, model
weights, or network, and they do not need torch or transformers: `encoder.SiglipEncoder` imports
both lazily.

```bash
cd services/image
pip install fastapi uvicorn pillow numpy pytest httpx
CI=1 python -m pytest -q -p no:cacheprovider
```

## Layout

- `app.py`: settings, request schema, error mapping, routes, and `main()`.
- `encoder.py`: the `Encoder` protocol, softmax scoring, the text-embedding LRU, `Classifier`, and
  the real `SiglipEncoder`.
- `imaging.py`: base64 and size limits, safe image decoding, and `pad_to_square`.
