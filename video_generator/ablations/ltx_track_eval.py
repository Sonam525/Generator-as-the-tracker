# Databricks notebook source
# PHASE-2 TRACKING eval on step-1500: IDF1 / IDSW / MOTA over the 25 held-out val videos
# (49-frame windows), reusing the training's step-1500 validation generations (right-half id-maps)
# and gentrack.eval.mot_metrics. Pred tracklets = decoded colours; GT = val gt.txt scaled to 512x512
# (the square eval space). HOTA/AssA need TrackEval (separate P4 step); IDF1/IDSW are the primary
# identity metrics available now. 49-frame windows => short-horizon (no long-gap re-acq) but real.

import json
import os
import subprocess
import sys
import time
import glob


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "motmetrics"], check=True)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval import mot_metrics  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
SAMP = f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase2/samples"
VALROOT = f"{VOL}/dancetrack/val"
RES = f"{VOL}/results/ltx_track_eval_step1500"
os.makedirs(RES, exist_ok=True)
K = 48
palette = build(K)
STEP = 1500
# validation-sample order in the phase2 config (idx -> val video)
VIDS = ["dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
        "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
        "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
        "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
        "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097"]


def right_half(d):
    out = []
    for p in sorted(glob.glob(f"{d}/*.png")):
        f = np.asarray(Image.open(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        out.append(f)
    return out


per, agg = {}, {"idf1@0.5": [], "idf1@0.3": [], "idsw@0.5": [], "mota@0.5": [], "gt_tracks": []}
for idx, vid in enumerate(VIDS, start=1):
    gen = right_half(f"{SAMP}/step_{STEP:06d}_{idx}_png")
    if not gen:
        log(f"{vid}: NO generation (skip)")
        continue
    n = len(gen)
    pred = idmaps_to_boxes(gen, palette=palette, min_area=150)     # frame 1..n, track_id = colour
    seq = load_sequence(f"{VALROOT}/{vid}")
    w, h = seq.image_size
    sx, sy = 512.0 / w, 512.0 / h
    gt = []
    for f in range(1, n + 1):
        for b in seq.boxes_at(f):
            gt.append(type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy,
                              b.conf, b.cls, b.vis))
    m05 = mot_metrics.evaluate(gt, pred, iou_threshold=0.5)
    m03 = mot_metrics.evaluate(gt, pred, iou_threshold=0.3)
    per[vid] = {"idf1@0.5": round(m05["idf1"], 3), "idf1@0.3": round(m03["idf1"], 3),
                "idsw@0.5": int(m05["num_switches"]), "mota@0.5": round(m05["mota"], 3),
                "gt_tracks": len({b.track_id for b in gt}), "pred_tracks": len({b.track_id for b in pred}),
                "frames": n}
    agg["idf1@0.5"].append(m05["idf1"]); agg["idf1@0.3"].append(m03["idf1"])
    agg["idsw@0.5"].append(m05["num_switches"]); agg["mota@0.5"].append(m05["mota"])
    agg["gt_tracks"].append(per[vid]["gt_tracks"])
    log(f"{vid}: IDF1@0.5 {per[vid]['idf1@0.5']} IDF1@0.3 {per[vid]['idf1@0.3']} "
        f"IDSW {per[vid]['idsw@0.5']} MOTA {per[vid]['mota@0.5']} (gt {per[vid]['gt_tracks']} tracks)")

out = {"checkpoint": f"step_{STEP}", "n_videos": len(agg["idf1@0.5"]),
       "mean_idf1@0.5": round(float(np.mean(agg["idf1@0.5"])), 3),
       "mean_idf1@0.3": round(float(np.mean(agg["idf1@0.3"])), 3),
       "mean_mota@0.5": round(float(np.mean(agg["mota@0.5"])), 3),
       "total_idsw@0.5": int(np.sum(agg["idsw@0.5"])),
       "mean_gt_tracks": round(float(np.mean(agg["gt_tracks"])), 1),
       "note": "49-frame windows; HOTA/AssA require TrackEval (separate)", "per_video": per}
with open(f"{RES}/metrics.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"=== mean IDF1@0.5 {out['mean_idf1@0.5']} | IDF1@0.3 {out['mean_idf1@0.3']} | "
    f"MOTA@0.5 {out['mean_mota@0.5']} | total IDSW {out['total_idsw@0.5']} over {out['n_videos']} videos ===")
log(f"DONE -> {RES}/metrics.json")
