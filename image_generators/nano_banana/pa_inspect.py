"""Inspect NB2 raw ID-map output on real dance frames (perceive + one edit).
Decodes FREE (no fixed assignment) to see what colours NB2 actually uses, then
canonicalises and tests whether edit() propagates those canonical colours.
Saves PNGs for visual review. Costs ~$0.30."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from gentrack.color_codec import decode, encode  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval.run_pA import load_frames  # noqa: E402
from gentrack.models.nanobanana_api import NanoBananaAdapter  # noqa: E402
from gentrack.palette import assign, build  # noqa: E402
from gentrack.pipeline.memory_canvas import MemoryCanvas  # noqa: E402
from gentrack.pipeline.track import reencode  # noqa: E402

OUT = Path("pa_scratch/inspect")
OUT.mkdir(parents=True, exist_ok=True)
MIN = 300
K = 32
palette = build(K)

seq = load_sequence(sys.argv[1])
frames = load_frames(seq, 2)
adapter = NanoBananaAdapter()


def save(name, arr):
    Image.fromarray(np.asarray(arr, np.uint8)).save(OUT / name)


def summarize(label, raw):
    free = decode(raw, palette=palette, assignment=None, min_area=MIN)
    areas = {int(k): int(v.sum()) for k, v in sorted(free.items())}
    print(f"[{label}] free-decode palette_idx->area: {areas}", flush=True)
    return free


# 1) perceive frame 1
raw1 = adapter.perceive(frames[0])
save("f1_frame.png", frames[0])
save("f1_raw.png", raw1)
free1 = summarize("perceive f1", raw1)

# canonicalise: largest blobs -> ids 0..k-1 -> our palette colours
det = sorted(free1.items(), key=lambda kv: -int(kv[1].sum()))
canon_ids = list(range(len(det)))
masks = [m for _, m in det]
if masks:
    canon_assign = assign(canon_ids, K)
    clean1 = encode(masks, canon_ids, palette=palette, assignment=canon_assign)
else:
    canon_assign = assign([0], K)
    clean1 = reencode({}, frames[0].shape[:2], palette, canon_assign)
save("f1_canonical.png", clean1)

# 2) edit frame 2 conditioned on the canonical map (full Option-A canvas)
if masks:
    canvas = MemoryCanvas(palette=palette, assignment=canon_assign)
    canvas.update(frames[0], {i: m for i, m in zip(canon_ids, masks)})
    cond = canvas.conditioning(frames[1], frames[0], clean1)
    raw2 = adapter.edit(frames[1], cond)
    save("f2_frame.png", frames[1])
    save("f2_raw.png", raw2)
    summarize("edit f2", raw2)
    print(f"canonical palette indices seeded: {sorted(canon_assign.values())}", flush=True)

print(f"total cost ${adapter.cost_usd:.2f}  -> saved in {OUT}", flush=True)
