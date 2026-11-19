"""One-shot NB2 contract smoke test. Costs ~$0.15 on success; 400s are free.

Verifies the live Gemini image API end-to-end through NanoBananaAdapter:
request build (responseModalities), retry path, and image parsing. Saves the
input frame and the returned ID-map to smoke_out/ for visual inspection.

Run:  $env:GEMINI_API_KEY=<key>;  python image_generators/nano_banana/smoke_nb2.py
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from gentrack.color_codec import decode  # noqa: E402
from gentrack.models.nanobanana_api import NanoBananaAdapter  # noqa: E402


def make_frame() -> np.ndarray:
    """Gray scene with three distinct colored blobs (stand-in 'people')."""
    h, w = 384, 512
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img[120:300, 60:160] = (200, 60, 60)
    img[100:320, 220:330] = (60, 160, 200)
    img[140:280, 380:460] = (80, 200, 90)
    return img


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: set GEMINI_API_KEY first")
        return 2

    adapter = NanoBananaAdapter()
    frame = make_frame()
    out_dir = Path(__file__).resolve().parents[2] / "smoke_out"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(frame).save(out_dir / "frame.png")

    try:
        out = adapter.perceive(frame)
    except Exception as e:  # noqa: BLE001 — surface the raw failure for diagnosis
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1

    Image.fromarray(out).save(out_dir / "nb2_idmap.png")
    rec = decode(out)
    print(f"OK  output_shape={out.shape}  cost=${adapter.cost_usd:.2f}")
    print(f"decoded_instances={len(rec)}  (palette-quantized, raw NB2 output)")
    print(f"saved: {out_dir/'frame.png'} , {out_dir/'nb2_idmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
