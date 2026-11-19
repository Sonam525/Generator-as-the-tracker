# Databricks notebook source
# PHASE-2 GENERALIZATION eval: decode the IC-LoRA's validation generations for 3 HELD-OUT clean
# val videos (dancetrack0004/0005/0007 — never in the 40-video training set), across checkpoints
# 500..3000. Right-half only (validation output is the [reference | generated] composite).
# color_iou/fg_iou vs GT filled-box id-maps. Climbing + high final => generalizes to unseen videos.

import glob
import json
import os
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy"], check=True)
import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
SAMP = f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase2/samples"
VGT = f"{VOL}/ltx_ds/phase2_val/idmap"
RES = f"{VOL}/results/ltx_p2_eval"
os.makedirs(RES, exist_ok=True)
K = 48
palette = build(K)
# validation sample order in the phase2 config -> held-out val clip
VAL = [(1, "dancetrack0004_000001_window"), (2, "dancetrack0005_000001_window"),
       (3, "dancetrack0007_000001_window")]
STEPS = [500, 1000, 1500, 2000, 2500, 3000]


def right_half(frames):
    out = []
    for f in frames:
        f = np.asarray(f)[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        out.append(f)
    return out


def read_png_dir(d):
    ps = sorted(glob.glob(f"{d}/*.png"))
    return right_half([Image.open(p) for p in ps]) if ps else []


def read_mkv(p):
    r = iio.get_reader(p); fr = [np.asarray(f)[:, :, :3] for f in r]; r.close(); return fr


def box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def mean_iou(pred, gt):
    if not gt:
        return None
    if not pred:
        return 0.0
    C = np.array([[box_iou(g.xywh, p.xywh) for p in pred] for g in gt])
    r, c = linear_sum_assignment(-C)
    return float(np.mean([C[i, j] for i, j in zip(r, c)]))


def clip_iou(gen, gt):
    n = min(len(gen), len(gt))
    vals = [mean_iou(idmaps_to_boxes([gen[i]], palette=palette, min_area=150),
                     idmaps_to_boxes([gt[i]], palette=palette, min_area=150)) for i in range(n)]
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 3) if vals else 0.0


def fg_iou(gen, gt):
    n = min(len(gen), len(gt))
    v = []
    for i in range(n):
        a = gen[i].sum(2) > 30; b = gt[i].sum(2) > 30; u = (a | b).sum()
        if u > 0:
            v.append(float((a & b).sum()) / float(u))
    return round(float(np.mean(v)), 3) if v else 0.0


GTs = {clip: read_mkv(f"{VGT}/{clip}.mkv") for _, clip in VAL}
out = {"val_clips": [c for _, c in VAL], "trajectory": {}}
for step in STEPS:
    row = {}
    for idx, clip in VAL:
        gen = read_png_dir(f"{SAMP}/step_{step:06d}_{idx}_png")
        gt = GTs[clip]
        if not gen:
            continue
        row[clip] = {"color_iou": clip_iou(gen, gt), "fg_iou": fg_iou(gen, gt),
                     "bg": round(float((np.stack(gen).reshape(-1, 3).astype(int).sum(1) < 30).mean()), 3)}
    if row:
        row["mean_color_iou"] = round(float(np.mean([v["color_iou"] for v in row.values()
                                                     if isinstance(v, dict)])), 3)
        row["mean_fg_iou"] = round(float(np.mean([v["fg_iou"] for v in row.values()
                                                  if isinstance(v, dict)])), 3)
    out["trajectory"][step] = row
    log(f"step {step}: mean_color_iou {row.get('mean_color_iou')} mean_fg_iou {row.get('mean_fg_iou')}")

with open(f"{RES}/metrics.json", "w") as f:
    json.dump(out, f, indent=2)

# contact sheet at the final step: rows=val videos, cols=[GT | gen]
H = 240
def rz(a):
    return np.asarray(Image.fromarray(a.astype(np.uint8)).resize((H, H)))
rows = []
for idx, clip in VAL:
    gen = read_png_dir(f"{SAMP}/step_003000_{idx}_png")
    gt = GTs[clip]
    if not gen:
        continue
    m = min(len(gen), len(gt)) // 2
    rows.append(np.concatenate([rz(gt[m]), rz(gen[m])], axis=1))
if rows:
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(f"{RES}/contact.png")
log(f"DONE -> {RES} (contact rows: 3 val videos, cols: GT | generated)")
