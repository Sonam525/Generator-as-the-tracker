"""Instrumented P-A diagnostic: 10-frame canonical-tracker loop with canvas +
crisp conditioning. Logs per-frame detections/matching and saves raw+canonical
maps so we can see whether churn is from NB2 colour drift or from matching.
Usage: python image_generators/nano_banana/pa_diag2.py <seq_dir> <n>"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from gentrack.color_codec import decode  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval.run_pA import load_frames  # noqa: E402
from gentrack.models.nanobanana_api import NanoBananaAdapter  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.canonicalize import CanonicalTracker  # noqa: E402
from gentrack.pipeline.memory_canvas import MemoryCanvas  # noqa: E402
from gentrack.pipeline.track import reencode  # noqa: E402

OUT = Path("pa_scratch/diag2")
OUT.mkdir(parents=True, exist_ok=True)
K = 32
palette = build(K)

seq = load_sequence(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 15000  # 1080p: ~0.7% of frame
frames = load_frames(seq, N)
adapter = NanoBananaAdapter()
tracker = CanonicalTracker(palette, K=K, min_area=MIN)
canvas = MemoryCanvas(palette=palette)


def save(name, arr):
    Image.fromarray(np.asarray(arr, np.uint8)).save(OUT / name)


prev_clean = prev_frame = None
for i, frame in enumerate(frames):
    if prev_clean is None:
        raw = adapter.perceive(frame)
    else:
        refs = [prev_frame, prev_clean, canvas.build_canvas()]  # prev photo, crisp prev map, swatches
        raw = adapter.edit(frame, refs)

    dets = {int(k): int(v.sum()) for k, v in decode(raw, palette=palette, assignment=None, min_area=MIN).items()}
    vis = tracker.update(raw, frame)
    assignment = tracker.assignment()
    clean = reencode(vis, frame.shape[:2], palette, assignment)
    canvas.assignment = assignment
    canvas.update(frame, vis)

    save(f"{i:02d}_raw.png", raw)
    save(f"{i:02d}_clean.png", clean)
    print(f"f{i}: NB2_dets(idx->area)={dets} | visible_ids={sorted(vis)} "
          f"| total_tracks={len(tracker.tracks)} | assign={assignment}", flush=True)

    prev_clean, prev_frame = clean, frame

print(f"cost ${adapter.cost_usd:.2f}  saved -> {OUT}", flush=True)
