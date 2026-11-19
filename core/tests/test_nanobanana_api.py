"""NB2 adapter: request build, response parse, retry/backoff, cost — all WITHOUT
network or spend (the single `_post` network method is monkeypatched)."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from gentrack.models.nanobanana_api import (
    MAX_REFERENCE_IMAGES,
    NanoBananaAdapter,
    NanoBananaError,
    TransientAPIError,
    estimate_cost,
)


def _fake_image_response(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "candidates": [
            {"content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": b64}}]}}
        ]
    }


def _adapter(**kw):
    kw.setdefault("base_delay", 0.0)  # no real sleeping in tests
    return NanoBananaAdapter(api_key="test-key", **kw)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(NanoBananaError):
        NanoBananaAdapter()


def test_build_request_structure():
    a = _adapter()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    body = a._build_request("hello", [frame, frame])
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "hello"}
    assert len(parts) == 3  # text + 2 images
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]


def test_perceive_roundtrips_image(monkeypatch):
    a = _adapter()
    want = np.dstack([np.full((6, 6), v, np.uint8) for v in (10, 20, 30)])
    monkeypatch.setattr(a, "_post", lambda body: _fake_image_response(want))
    out = a.perceive(np.zeros((6, 6, 3), dtype=np.uint8))
    assert np.array_equal(out, want)
    assert a.images_generated == 1
    assert a.cost_usd == pytest.approx(0.15)


def test_edit_accepts_multiple_refs(monkeypatch):
    a = _adapter()
    seen = {}
    want = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_post(body):
        seen["n"] = len(body["contents"][0]["parts"])
        return _fake_image_response(want)

    monkeypatch.setattr(a, "_post", fake_post)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    a.edit(frame, [frame, frame, frame])  # frame + 3 refs
    assert seen["n"] == 1 + 4  # text + (frame + 3 refs)


def test_too_many_images():
    a = _adapter()
    imgs = [np.zeros((2, 2, 3), dtype=np.uint8)] * (MAX_REFERENCE_IMAGES + 1)
    with pytest.raises(NanoBananaError):
        a.generate(imgs, "x")


def test_retry_then_success(monkeypatch):
    a = _adapter(max_retries=5)
    want = np.zeros((4, 4, 3), dtype=np.uint8)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientAPIError("HTTP 503")
        return _fake_image_response(want)

    monkeypatch.setattr(a, "_post", flaky)
    monkeypatch.setattr("gentrack.models.nanobanana_api.time.sleep", lambda s: None)
    out = a.generate([want], "x")
    assert calls["n"] == 3 and np.array_equal(out, want)


def test_retry_exhausted(monkeypatch):
    a = _adapter(max_retries=3)
    monkeypatch.setattr(a, "_post", lambda body: (_ for _ in ()).throw(TransientAPIError("503")))
    monkeypatch.setattr("gentrack.models.nanobanana_api.time.sleep", lambda s: None)
    with pytest.raises(NanoBananaError):
        a.generate([np.zeros((2, 2, 3), dtype=np.uint8)], "x")


def test_parse_response_no_image():
    a = _adapter()
    with pytest.raises(NanoBananaError):
        a._parse_response({"candidates": [{"content": {"parts": [{"text": "nope"}]}}]})


def test_output_resized_to_frame(monkeypatch):
    """NB2 returns its own resolution; perceive/edit must resize back to the frame."""
    a = _adapter()
    # NB2 "returns" a 30x40 image (HxW) regardless of the 12x16 input frame
    big = np.zeros((30, 40, 3), dtype=np.uint8)
    big[:, :20] = (200, 50, 50)  # flat colour block, must survive NEAREST resize
    monkeypatch.setattr(a, "_post", lambda body: _fake_image_response(big))
    frame = np.zeros((12, 16, 3), dtype=np.uint8)  # H=12, W=16
    out = a.perceive(frame)
    assert out.shape == (12, 16, 3)  # resized to the frame
    # only flat palette-ish colours present (NEAREST introduces no blends)
    colours = {tuple(c) for c in out.reshape(-1, 3)}
    assert colours <= {(0, 0, 0), (200, 50, 50)}


def test_estimate_cost():
    assert estimate_cost(100) == pytest.approx(15.0)
