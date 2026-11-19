"""Nano Banana 2 (Gemini image API) adapter — the ONLY path to NB2.

Closed, paid API (~$0.15/image). Preview is 503-prone, so this client owns the
retry/backoff. Supports up to 14 reference images — the zero-training memory bank
for Option A (pipeline/memory_canvas.py).

Spend guardrails live in the runner, not here: estimate cost and confirm with the
human above ~$10 before any batch; never run over a full split. This adapter just
tracks `images_generated` / `cost_usd` so the runner can enforce that.

Prompts are experimental parameters, pinned in image_generators/nano_banana/nanobanana_api.yaml and
mirrored here as defaults — keep the two in sync. They follow the §4 probe rules:
pair RGB with ID-maps, demand flat colours on black, and NEVER ask the model to
"reason" about identity across frames (it hallucinates).

The network call is isolated in `_post` so tests patch it — no spend, no network.
"""
from __future__ import annotations

import base64
import io
import os
import random
import time

import numpy as np
from PIL import Image

from gentrack.models.base import GeneratorAdapter

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
PRICE_PER_IMAGE_USD = 0.15
MAX_REFERENCE_IMAGES = 14

# Configurable HARM categories set permissive — DanceTrack dance imagery trips the
# default filter. NOTE: the non-configurable PROHIBITED_CONTENT filter cannot be
# disabled this way; if blocks persist with these set, the content is hard-blocked.
DEFAULT_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

PERCEIVE_PROMPT = (
    "Produce a segmentation ID-map of this image. Paint every distinct person "
    "as a single flat, solid colour on a pure black background. Use a clearly "
    "different colour for each person. No gradients, no outlines, no text, no "
    "shading — flat colour fills only."
)

EDIT_PROMPT = (
    "The reference images show the previous frame, its ID-map (each person a flat "
    "colour on black), and optionally a memory panel of known people with their "
    "colours. Produce the ID-map for the NEW (first) image. Keep each person's "
    "colour EXACTLY as in the previous ID-map / memory panel. Give a newly "
    "appearing person a new unused flat colour. Pure black background, flat colour "
    "fills only — no gradients, outlines, text, or shading. Do not guess identities."
)


class NanoBananaError(RuntimeError):
    """Non-retryable API failure."""


class TransientAPIError(NanoBananaError):
    """Retryable failure (HTTP 429/503)."""


class NanoBananaBlocked(NanoBananaError):
    """Content blocked by the API safety filter (not retryable)."""


def _img_to_b64(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_img(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _wh(frame: np.ndarray) -> tuple[int, int]:
    """(width, height) of an (H, W, 3) frame."""
    return frame.shape[1], frame.shape[0]


def estimate_cost(n_images: int) -> float:
    """USD for generating n_images. Use in runners to gate batches."""
    return n_images * PRICE_PER_IMAGE_USD


class NanoBananaAdapter(GeneratorAdapter):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 45.0,
        timeout: float = 120.0,
        perceive_prompt: str = PERCEIVE_PROMPT,
        edit_prompt: str = EDIT_PROMPT,
        response_modalities: tuple[str, ...] = ("IMAGE",),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise NanoBananaError("no API key: pass api_key or set GEMINI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.perceive_prompt = perceive_prompt
        self.edit_prompt = edit_prompt
        self.response_modalities = response_modalities
        self.safety_settings = safety_settings
        self.images_generated = 0

    @property
    def cost_usd(self) -> float:
        return self.images_generated * PRICE_PER_IMAGE_USD

    # ---- GeneratorAdapter interface ----
    def perceive(self, frame: np.ndarray) -> np.ndarray:
        return self.generate([frame], self.perceive_prompt, out_size=_wh(frame))

    def edit(self, frame: np.ndarray, cond) -> np.ndarray:
        refs = list(cond) if isinstance(cond, (list, tuple)) else [cond]
        return self.generate([frame, *refs], self.edit_prompt, out_size=_wh(frame))

    # ---- core ----
    def generate(self, images, prompt: str, out_size: tuple[int, int] | None = None) -> np.ndarray:
        """Generate one ID-map. `out_size` = (width, height): NB2 returns its own
        resolution, so we resize back with NEAREST to keep flat colours decodable.
        """
        images = list(images)
        if not images:
            raise NanoBananaError("generate needs at least one image")
        if len(images) > MAX_REFERENCE_IMAGES:
            raise NanoBananaError(
                f"{len(images)} images exceed the {MAX_REFERENCE_IMAGES}-image limit"
            )
        body = self._build_request(prompt, images)
        resp = self._post_with_retry(body)
        img = self._parse_response(resp)
        if out_size is not None and (img.shape[1], img.shape[0]) != tuple(out_size):
            img = np.asarray(
                Image.fromarray(img, "RGB").resize(tuple(out_size), Image.NEAREST),
                dtype=np.uint8,
            )
        self.images_generated += 1
        return img

    def _endpoint(self) -> str:
        return f"{_API_BASE}/{self.model}:generateContent"

    def _build_request(self, prompt: str, images) -> dict:
        parts = [{"text": prompt}]
        for im in images:
            parts.append(
                {"inline_data": {"mime_type": "image/png", "data": _img_to_b64(im)}}
            )
        body: dict = {"contents": [{"parts": parts}]}
        if self.response_modalities:
            body["generationConfig"] = {"responseModalities": list(self.response_modalities)}
        if self.safety_settings:
            body["safetySettings"] = self.safety_settings
        return body

    def _parse_response(self, resp: dict) -> np.ndarray:
        cands = resp.get("candidates")
        if not cands:
            reason = (resp.get("promptFeedback") or {}).get("blockReason")
            if reason:
                raise NanoBananaBlocked(f"prompt blocked: {reason}")
            raise NanoBananaError(f"malformed response (no candidates): {resp}")
        cand = cands[0]
        for p in (cand.get("content") or {}).get("parts") or []:
            data = p.get("inline_data") or p.get("inlineData")
            if data and data.get("data"):
                return _b64_to_img(data["data"])
        fr = cand.get("finishReason")
        if fr and fr not in ("STOP", "MAX_TOKENS"):
            raise NanoBananaBlocked(f"no image returned; finishReason={fr}")
        raise NanoBananaError(f"response has no image part: {cand}")

    def _post(self, body: dict) -> dict:
        """Single HTTP POST. Patched in tests — keep network strictly here."""
        import requests  # lazy: tests never hit this path

        resp = requests.post(
            self._endpoint(),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            raise TransientAPIError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise NanoBananaError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _post_with_retry(self, body: dict) -> dict:
        delay = self.base_delay
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._post(body)
            except TransientAPIError as e:
                last = e
                if attempt == self.max_retries - 1:
                    break
                # full-jitter exponential backoff
                time.sleep(min(self.max_delay, delay) * (0.5 + random.random()))
                delay *= 2
        raise NanoBananaError(f"exhausted {self.max_retries} retries; last: {last}")
