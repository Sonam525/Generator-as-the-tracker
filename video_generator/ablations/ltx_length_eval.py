# Databricks notebook source
# Path-B length test (BOX-level, the reliable metric): decode the 121-frame generation (right half)
# via idmaps_to_boxes (component-level palette match, robust) and score IDF1/IDSW/MOTA vs DanceTrack GT
# over frame ranges 1-49 / 1-121 / 50-121. Question: does identity hold PAST the 49f training horizon?
# (Per-pixel palette IoU is unreliable here: val GT = filled boxes vs generated silhouettes, and
#  per-pixel palette snapping fragments colours -- so we compare at the box level like the 49f eval.)

import glob
import json
import os
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy", "motmetrics"], check=True)
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval import mot_metrics  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
GEN = f"{VOL}/checkpoints/ltx_v2v_ic_lora_lengthtest/samples/step_001500_1_png"
VID = "dancetrack0065"
RES = f"{VOL}/results/ltx_lengthtest_0065_L121"
os.makedirs(RES, exist_ok=True)
K = 48
palette = build(K)

gen = []
for p in sorted(glob.glob(f"{GEN}/*.png")):
    a = np.asarray(imageio.imread(p))[:, :, :3]
    if a.shape[1] == 2 * a.shape[0]:
        a = a[:, a.shape[1] // 2:, :]
    gen.append(a)
n = len(gen)
pred = idmaps_to_boxes(gen, palette=palette, min_area=150)          # frame 1..n, track_id = colour idx
seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w, h = seq.image_size
sx, sy = 512.0 / w, 512.0 / h                                       # GT -> 512 square eval space
log(f"gen frames {n} | pred boxes {len(pred)} | GT {w}x{h}")


def gt_in(a, b):
    out = []
    for f in range(a, b + 1):
        for bx in seq.boxes_at(f):
            out.append(type(bx)(f, bx.track_id, bx.x * sx, bx.y * sy, bx.w * sx, bx.h * sy,
                                bx.conf, bx.cls, bx.vis))
    return out


def pred_in(a, b):
    return [p for p in pred if a <= p.frame <= b]


res = {}
for lbl, a, b in [("f1_49", 1, 49), ("f1_121", 1, n), ("f50_121", 50, n)]:
    gt, pr = gt_in(a, b), pred_in(a, b)
    m5 = mot_metrics.evaluate(gt, pr, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt, pr, iou_threshold=0.3)
    res[lbl] = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
                    idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
                    gt_tracks=len({x.track_id for x in gt}),
                    pred_tracks=len({x.track_id for x in pr}), frames=b - a + 1)
    log(f"{lbl:<9} IDF1@0.5 {res[lbl]['idf1_05']:<6} IDF1@0.3 {res[lbl]['idf1_03']:<6} "
        f"IDSW {res[lbl]['idsw']:<4} MOTA {res[lbl]['mota_05']:<7} "
        f"(gt {res[lbl]['gt_tracks']} / pred {res[lbl]['pred_tracks']} tracks)")

out = dict(video=VID, length=n, trained_horizon=49, iou_space="512x512", ranges=res,
           note="box-level IDF1; compares identity at trained (1-49) vs extended (50-121) horizon")
with open(f"{RES}/metrics_boxlevel.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_boxlevel.json")
