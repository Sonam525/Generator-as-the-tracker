"""Evaluate saved canonical clean maps against GT (box-IoU MOT metrics). No API.
Usage: python image_generators/nano_banana/pa_eval.py <clean_maps_dir> <seq_dir> <n_frames>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from gentrack.color_codec import load_idmap  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval.mot_metrics import evaluate  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

clean_dir = Path(sys.argv[1])
seq_dir = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 9999

palette = build(32)
clean_files = sorted(clean_dir.glob("*_clean.png"))[:N]
clean_maps = [load_idmap(f) for f in clean_files]
pred = idmaps_to_boxes(clean_maps, palette=palette)

seq = load_sequence(seq_dir)
gt = [b for f in range(1, len(clean_maps) + 1) for b in seq.boxes_at(f)]

m = evaluate(gt, pred, iou_threshold=0.5)
print(f"frames={len(clean_maps)}  gt_dets={len(gt)}  pred_dets={len(pred)}  "
      f"gt_tracks={len({b.track_id for b in gt})}  pred_tracks={len({b.track_id for b in pred})}")
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, indent=2))
