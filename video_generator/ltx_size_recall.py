# Databricks notebook source
# Dense-scene detection diagnosis (CPU, reuses Arm-A cleaned windows -- no GPU).
# For the worst dense videos + sparse controls: per-frame detection recall of GT people binned by
# GT box height (512 space), plus a MERGE rate (one pred box covering 2+ GT people at IoU>=0.3) and
# per-frame painted-vs-GT count. Separates the two hypotheses: "small people vanish at 512^2" vs
# "crowded same-costume people merge".

import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy"], check=True)
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
try:
    ARM = dbutils.widgets.get("arm_dir") or "ltx_arm_a_chain"      # noqa: F821
    SUFFIX = dbutils.widgets.get("win_suffix") or "_ov9"           # noqa: F821
    SIZE = int(dbutils.widgets.get("size") or "512")               # noqa: F821
    VIDS_IN = dbutils.widgets.get("vids") or ""                    # noqa: F821
except Exception:
    ARM, SUFFIX, SIZE, VIDS_IN = "ltx_arm_a_chain", "_ov9", 512, ""
# 5 worst dense + 2 sparse controls (default)
VIDS = [v.strip() for v in VIDS_IN.split(",") if v.strip()] or        ["dancetrack0041", "dancetrack0026", "dancetrack0081", "dancetrack0094", "dancetrack0014",
        "dancetrack0097", "dancetrack0018"]
RES = f"{VOL}/results/ltx_dense_diagnosis"
OUT_TAG = f"{ARM}{SUFFIX}"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, MATCH_IOU = 48, 150, 0.3
BINS = [(0, 40, "small<40"), (40, 80, "mid40-80"), (80, 10000, "large>=80")]


def iou_t(a, b):
    ix, iy = max(a[0], b[0]), max(a[1], b[1])
    iX = min(a[0] + a[2], b[0] + b[2]); iY = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a[2] * a[3] + b[2] * b[3] - inter
    return inter / uni if uni > 0 else 0.0


def process_video(vid):
    palette = build(K)
    meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{vid}{SUFFIX}/windows.json"))
    wins = sorted(meta["windows"], key=lambda w: w["idx"])
    total = meta["total_frames"]
    own_start = [wins[0]["start"]]
    for k in range(1, len(wins)):
        own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
    own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]

    pred_by_frame = {}
    for wi, w in enumerate(wins):
        frames = [np.asarray(imageio.imread(p))[:, :, :3]
                  for p in sorted(glob.glob(f"{VOL}/checkpoints/{ARM}/{vid}/cleaned/w{wi:02d}/*.png"))]
        if not frames:
            return dict(vid=vid, error=f"no frames w{wi:02d}")
        for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
            gf = w["start"] + (b.frame - 1)
            if own_start[wi] <= gf <= own_end[wi]:
                pred_by_frame.setdefault(gf, []).append((b.x, b.y, b.w, b.h))

    seq = load_sequence(f"{VOL}/dancetrack/val/{vid}")
    w0, h0 = seq.image_size
    sx, sy = float(SIZE) / w0, float(SIZE) / h0
    binstat = {lbl: [0, 0] for _, _, lbl in BINS}      # matched, total
    merges = 0
    npred_tot = ngt_tot = nframes = 0
    for f in range(1, total + 1):
        gts = [(b.x * sx, b.y * sy, b.w * sx, b.h * sy) for b in seq.boxes_at(f)]
        preds = pred_by_frame.get(f, [])
        nframes += 1; npred_tot += len(preds); ngt_tot += len(gts)
        gt_hit = [False] * len(gts)
        for p in preds:
            hits = [i for i, g in enumerate(gts) if iou_t(p, g) >= MATCH_IOU]
            if len(hits) >= 2:
                merges += 1
            for i in hits:
                gt_hit[i] = True
        for i, g in enumerate(gts):
            h = g[3]
            for lo, hi, lbl in BINS:
                if lo <= h < hi:
                    binstat[lbl][1] += 1
                    if gt_hit[i]:
                        binstat[lbl][0] += 1
                    break
    out = dict(vid=vid, frames=total,
               gt_per_frame=round(ngt_tot / nframes, 1), pred_per_frame=round(npred_tot / nframes, 1),
               merge_rate=round(merges / max(1, npred_tot), 3))
    for lbl, (m, t) in binstat.items():
        out[f"recall_{lbl}"] = round(m / t, 3) if t else None
        out[f"n_{lbl}"] = t
    return out


log(f"diagnosing {len(VIDS)} videos (match IoU {MATCH_IOU})")
with Pool(4) as pool:
    results = pool.map(process_video, VIDS)
for r in results:
    if "error" in r:
        log(f"{r['vid']}: ERROR {r['error']}"); continue
    log(f"{r['vid']}: gt/f {r['gt_per_frame']:>4} pred/f {r['pred_per_frame']:>4} | "
        f"recall small {r['recall_small<40']} (n={r['n_small<40']}) "
        f"mid {r['recall_mid40-80']} (n={r['n_mid40-80']}) "
        f"large {r['recall_large>=80']} (n={r['n_large>=80']}) | merge_rate {r['merge_rate']}")
with open(f"{RES}/size_recall_{OUT_TAG}.json", "w") as f:
    json.dump([r for r in results], f, indent=2)
log(f"DONE -> {RES}/size_recall_{OUT_TAG}.json")
