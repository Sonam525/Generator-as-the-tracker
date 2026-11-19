"""P-A occlusion re-acquisition test: run the Option-A chain over a clip with a
mined disappear→reappear gap, then measure whether identity is re-bound.
Usage: python image_generators/nano_banana/pa_reacq.py <seq_dir> <out_dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.mot_format import write_tracks  # noqa: E402
from gentrack.eval.mot_metrics import reacquisition_check  # noqa: E402
from gentrack.eval.run_pA import confirm_spend, evaluate_clip, load_frames, run_pa_clip, sequence_gt_masks  # noqa: E402
from gentrack.models.nanobanana_api import NanoBananaAdapter, estimate_cost  # noqa: E402
from gentrack.palette import build  # noqa: E402

SEQ = sys.argv[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pa_scratch/reacq")
K, MIN_AREA = 32, 15000

seq = load_sequence(SEQ)
frames = load_frames(seq)
gt_masks = sequence_gt_masks(seq)[: len(frames)]
palette = build(K)

print(f"seq={seq.name} frames={len(frames)} est=${estimate_cost(len(frames)):.2f}", flush=True)
confirm_spend(len(frames), confirm_above=10.0, assume_yes=True)

adapter = NanoBananaAdapter()
res = run_pa_clip(
    adapter, frames, gt_masks, palette=palette, K=K, min_area=MIN_AREA,
    use_canvas=True, contact_dir=OUT / "contact", contact_every=11,
)
metrics, pred, gt = evaluate_clip(res["clean_maps"], seq, palette)
events = mine_occlusion(seq.gt, strategy="absence", min_gap=5, max_gap=40)
reacq = reacquisition_check(pred, gt, events)

OUT.mkdir(parents=True, exist_ok=True)
write_tracks(pred, OUT / "pred.txt")
(OUT / "results.json").write_text(json.dumps(
    {"n_tracks": res["n_tracks"], "cost_usd": res["cost_usd"], "metrics": metrics, "reacq": reacq},
    indent=2, default=str))

print(f"n_tracks={res['n_tracks']}  cost=${res['cost_usd']:.2f}", flush=True)
print("metrics:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}), flush=True)
print("REACQ:", json.dumps(reacq, default=str), flush=True)
print("contact_sheets:", res.get("contact_sheets"), flush=True)
