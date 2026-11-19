# Databricks notebook source
# Path-B stitch + full-sequence eval. Decode every generated 49f window of one video, then STITCH
# identity across window boundaries by matching colours on the overlap frames (Hungarian on aggregated
# box-IoU) into one global track over all N frames. Scores IDF1/IDSW/MOTA for three linkings:
#   raw_color  - use the model's palette colour as the id (cross-window link only by colour coincidence)
#   no_link    - every window's tracks are fresh (identity never carries across a boundary; IDSW ceiling)
#   stitched   - explicit overlap matching (the Path-B mechanism)
# The gap stitched-vs-raw_color is the value added by explicit cross-window association.

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy", "motmetrics"], check=True)
import imageio.v2 as imageio  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval import mot_metrics  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
try:
    VID = dbutils.widgets.get("vid") or "dancetrack0065"      # noqa: F821
    SAMP = dbutils.widgets.get("samp") or f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples"  # noqa: F821,E501
except Exception:
    VID = "dancetrack0065"
    SAMP = f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples"
RES = f"{VOL}/results/ltx_fullseq_{VID}"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA = 48, 150
palette = build(K)

meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{VID}/windows.json"))
wins = sorted(meta["windows"], key=lambda w: w["start"])
total = meta["total_frames"]
log(f"{VID}: {len(wins)} windows, {total} frames, overlap {meta['overlap']}")


def right_half(d):
    out = []
    for p in sorted(glob.glob(f"{d}/*.png")):
        f = np.asarray(imageio.imread(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        out.append(f)
    return out


# decode each window -> {global_frame: [Box(track_id=local colour)]}. window idx wi maps to sample dir
# step_001500_{wi+1}_png (validation-sample order == window order).
win_boxes = []
for wi, w in enumerate(wins):
    frames = right_half(f"{SAMP}/step_001500_{wi+1}_png")
    if not frames:
        log(f"window {wi}: NO frames at step_001500_{wi+1}_png -> abort")
        raise SystemExit(1)
    boxes = idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA)   # frame 1..len, id=colour
    perf = {}
    for b in boxes:
        gf = w["start"] + (b.frame - 1)
        perf.setdefault(gf, []).append(b)
    win_boxes.append(perf)

# ownership: each global frame belongs to exactly one window (split overlaps at the midpoint)
own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]


def iou(a, b):
    ax2, ay2, bx2, by2 = a.x + a.w, a.y + a.h, b.x + b.w, b.y + b.h
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    iX, iY = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a.w * a.h + b.w * b.h - inter
    return inter / uni if uni > 0 else 0.0


# ---- STITCH: remap each window's local colours to global ids via overlap matching ----
maps = []                      # maps[wi][local_colour] = global_id
next_gid = [0]


def fresh(colours):
    m = {}
    for c in colours:
        m[c] = next_gid[0]; next_gid[0] += 1
    return m


colours0 = sorted({b.track_id for bs in win_boxes[0].values() for b in bs})
maps.append(fresh(colours0))
for wi in range(1, len(wins)):
    prev = wi - 1
    ov = [f for f in range(wins[wi]["start"], wins[prev]["end"] + 1)
          if f in win_boxes[wi] and f in win_boxes[prev]]
    cur_cols = sorted({b.track_id for bs in win_boxes[wi].values() for b in bs})
    prev_cols = sorted(maps[prev].keys())
    m = {}
    if ov and cur_cols and prev_cols:
        C = np.zeros((len(cur_cols), len(prev_cols)))
        for f in ov:
            cb = {b.track_id: b for b in win_boxes[wi][f]}
            pb = {b.track_id: b for b in win_boxes[prev][f]}
            for i, cc in enumerate(cur_cols):
                if cc not in cb:
                    continue
                for j, pc in enumerate(prev_cols):
                    if pc in pb:
                        C[i, j] += iou(cb[cc], pb[pc])
        ri, ci = linear_sum_assignment(-C)
        for i, j in zip(ri, ci):
            if C[i, j] > 0.5:                       # need real overlap agreement to link
                m[cur_cols[i]] = maps[prev][prev_cols[j]]
    for c in cur_cols:
        if c not in m:
            m[c] = next_gid[0]; next_gid[0] += 1
    maps.append(m)

# ---- assemble the three global tracks over owned frames ----
def assemble(mode):
    out = []
    for wi, w in enumerate(wins):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for b in win_boxes[wi].get(gf, []):
                if mode == "stitched":
                    gid = maps[wi][b.track_id]
                elif mode == "raw_color":
                    gid = b.track_id
                else:                                # no_link
                    gid = wi * 1000 + b.track_id
                out.append(type(b)(gf, gid, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis))
    return out


seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w0, h0 = seq.image_size
sx, sy = 512.0 / w0, 512.0 / h0
gt = []
for f in range(1, total + 1):
    for b in seq.boxes_at(f):
        gt.append(type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis))

out = {"video": VID, "total_frames": total, "n_windows": len(wins),
       "overlap": meta["overlap"], "gt_tracks": len({b.track_id for b in gt}), "modes": {}}
for mode in ["no_link", "raw_color", "stitched"]:
    pred = assemble(mode)
    m5 = mot_metrics.evaluate(gt, pred, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt, pred, iou_threshold=0.3)
    out["modes"][mode] = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
                              idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
                              pred_tracks=len({b.track_id for b in pred}))
    log(f"{mode:<10} IDF1@0.5 {out['modes'][mode]['idf1_05']:<6} IDF1@0.3 {out['modes'][mode]['idf1_03']:<6} "
        f"IDSW {out['modes'][mode]['idsw']:<5} MOTA {out['modes'][mode]['mota_05']:<7} "
        f"pred_tracks {out['modes'][mode]['pred_tracks']}")

with open(f"{RES}/metrics.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics.json  (gt has {out['gt_tracks']} tracks over {total} frames)")
