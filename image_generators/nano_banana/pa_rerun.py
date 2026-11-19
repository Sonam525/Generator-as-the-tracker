"""Re-run a P-A clip through the CanonicalTracker and write contact sheets.
Usage: python image_generators/nano_banana/pa_rerun.py <seq_dir> <n_frames> <out_dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.mot_format import write_tracks  # noqa: E402
from gentrack.eval.run_pA import (  # noqa: E402
    confirm_spend,
    evaluate_clip,
    load_frames,
    run_pa_clip,
    sequence_gt_masks,
)
from gentrack.models.nanobanana_api import NanoBananaAdapter, estimate_cost  # noqa: E402
from gentrack.palette import build  # noqa: E402

SEQ = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("pa_scratch/rerun")
K, MIN_AREA = 32, 15000  # min_area ~0.7% of a 1080p frame

seq = load_sequence(SEQ)
frames = load_frames(seq, N)
gt = sequence_gt_masks(seq)[: len(frames)]
palette = build(K)

print(f"seq={seq.name} frames={len(frames)} est=${estimate_cost(len(frames)):.2f}", flush=True)
confirm_spend(len(frames), confirm_above=10.0, assume_yes=True)

adapter = NanoBananaAdapter()
res = run_pa_clip(
    adapter, frames, gt, palette=palette, K=K, min_area=MIN_AREA,
    use_canvas=True, contact_dir=OUT / "contact", contact_every=10,
)
OUT.mkdir(parents=True, exist_ok=True)

metrics, pred, gt_boxes = evaluate_clip(res["clean_maps"], seq, palette)
write_tracks(pred, OUT / "pred.txt")
write_tracks(gt_boxes, OUT / "gt.txt")
summary = {k: v for k, v in res.items() if k != "clean_maps"}
summary["metrics"] = metrics
(OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))

print(f"n_tracks={res['n_tracks']}  cost=${res['cost_usd']:.2f}", flush=True)
print("metrics:", json.dumps(
    {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}), flush=True)
print("contact_sheets:", res.get("contact_sheets"), flush=True)
