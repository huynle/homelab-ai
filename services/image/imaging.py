"""Image decoding and preprocessing. Pure Pillow: no torch, no model."""

from __future__ import annotations

import base64
import binascii
import io
import math
import warnings

from PIL import Image

PAD_FILL = (255, 255, 255)


class ImageError(ValueError):
    """The payload is not a usable image (maps to 400 invalid_request)."""


class ImageTooLarge(ValueError):
    """The decoded payload exceeds the byte limit (maps to 413 payload_too_large)."""


def max_base64_chars(max_bytes: int) -> int:
    return 4 * math.ceil(max_bytes / 3)


def decode_base64(data: str, max_bytes: int) -> bytes:
    """Decode standard base64, enforcing `max_bytes` on the decoded size."""
    if len(data) > max_base64_chars(max_bytes):
        approx = len(data) * 3 // 4
        raise ImageTooLarge(f"decoded image is about {approx} bytes; limit is {max_bytes}")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("image.base64 is not valid base64") from None
    if len(raw) > max_bytes:
        raise ImageTooLarge(f"decoded image is {len(raw)} bytes; limit is {max_bytes}")
    return raw


def load_image(raw: bytes, max_pixels: int) -> Image.Image:
    """Open, bound-check, fully decode, and convert to RGB."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(raw))
            width, height = img.size
            if width * height > max_pixels:
                raise ImageError(f"image is {width}x{height} pixels; limit is {max_pixels} pixels")
            img.load()
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError(f"image exceeds the {max_pixels} pixel limit") from None
    except Exception:
        raise ImageError("image.base64 does not decode to a supported image") from None
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        # Composite transparency onto the same white used for padding.
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, PAD_FILL)
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] = PAD_FILL) -> Image.Image:
    """Center `image` on a square canvas of side max(w, h) filled with `fill`.

    The SigLIP processor resizes straight to 224x224 without keeping aspect ratio, so tall
    receipts get squashed; padding first keeps the content's proportions.
    """
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas
