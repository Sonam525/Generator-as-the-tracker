# Databricks notebook source
# Eval the CHAINED (overlap-prefix-conditioned) windows vs the independent-window results.
# Reads the chain's CLEANED id-map windows (right-half already extracted + decode->correct applied),
# decodes to boxes, and scores the full sequence in two modes:
#   raw_color   - trust the model's colours globally (the chain's whole point: prefix conditioning
#                 should make colours consistent across windows BY CONSTRUCTION)
#   stitched_v3 - many-to-one overlap stitcher on top (belt-and-braces)
# Pre-registered comparison (independent windows, stride-41 set): raw_color 0.196 / v3 0.466 /
# oracle ceiling 0.698 / in-window mean 0.661. Success = chained raw_color >> 0.196.
# Also computes the chained ORACLE + per-window IDF1 so ceilings are comparable on this window set.

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter

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
    VID = dbutils.widgets.get("vid") or "dancetrack0065"          # noqa: F821
    WSUB = dbutils.widgets.get("windows_sub") or "dancetrack0065_ov9"  # noqa: F821
    CHAIN = dbutils.widgets.get("chain_out") or "ltx_chain_0065"  # noqa: F821
except Exception:
    VID, WSUB, CHAIN = "dancetrack0065", "dancetrack0065_ov9", "ltx_chain_0065"
CLEANED = f"{VOL}/checkpoints/{CHAIN}/cleaned"
RES = f"{VOL}/results/{CHAIN}"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, OVERLAP_LINK, ORACLE_MIN_CREDIT = 48, 150, 0.5, 0.5
palette = build(K)

meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{WSUB}/windows.json"))
wins = sorted(meta["windows"], key=lambda w: w["idx"])
total = meta["total_frames"]
log(f"{VID} chained: {len(wins)} windows, {total} frames, overlap {meta['overlap']}")


def load_cleaned(k):
    out = []
    for p in sorted(glob.glob(f"{CLEANED}/w{k:02d}/*.png")):
        out.append(np.asarray(imageio.imread(p))[:, :, :3])
    return out


def iou(a, b):
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    iX, iY = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a.w * a.h + b.w * b.h - inter
    return inter / uni if uni > 0 else 0.0


win_boxes, win_colours = [], []
for wi, w in enumerate(wins):
    frames = load_cleaned(wi)
    if not frames:
        log(f"window {wi}: NO cleaned frames -> abort"); raise SystemExit(1)
    perf = {}
    for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
        perf.setdefault(w["start"] + (b.frame - 1), {})[b.track_id] = b
    win_boxes.append(perf)
    win_colours.append(sorted({c for d in perf.values() for c in d}))

seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w0, h0 = seq.image_size
sx, sy = 512.0 / w0, 512.0 / h0
gt_by_frame = {f: [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy,
                           b.conf, b.cls, b.vis) for b in seq.boxes_at(f)]
               for f in range(1, total + 1)}
gt_all = [g for gs in gt_by_frame.values() for g in gs]

own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]

# per-window IDF1 (in-window level on the CHAINED set)
per_win = []
for wi, w in enumerate(wins):
    pred = [type(b)(gf, c, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis)
            for gf, d in win_boxes[wi].items() for c, b in d.items()]
    gtw = [g for f in range(w["start"], w["end"] + 1) for g in gt_by_frame[f]]
    per_win.append(round(mot_metrics.evaluate(gtw, pred, iou_threshold=0.5)["idf1"], 3))
log(f"per-window IDF1@0.5: mean {np.mean(per_win):.3f} | {per_win}")

# oracle on the chained set
oracle = {}
for wi in range(len(wins)):
    credit = {}
    for gf, d in win_boxes[wi].items():
        for c, b in d.items():
            for g in gt_by_frame.get(gf, []):
                v = iou(b, g)
                if v > 0:
                    credit.setdefault(c, Counter())[g.track_id] += v
    for c in win_colours[wi]:
        best = credit.get(c, Counter()).most_common(1)
        oracle[(wi, c)] = best[0][0] if best and best[0][1] > ORACLE_MIN_CREDIT else None


def assemble_raw():
    return [type(b)(gf, c, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis)
            for wi in range(len(wins)) for gf in range(own_start[wi], own_end[wi] + 1)
            for c, b in win_boxes[wi].get(gf, {}).items()]


def stitch_many():
    maps = [dict() for _ in wins]
    nid = 0
    for wi in range(len(wins)):
        cols = win_colours[wi]
        if wi > 0:
            prev = wi - 1
            ov = [f for f in range(wins[wi]["start"], wins[prev]["end"] + 1)
                  if f in win_boxes[wi] and f in win_boxes[prev]]
            pcols = win_colours[prev]
            C = np.zeros((len(cols), len(pcols)))
            for f in ov:
                cb, pb = win_boxes[wi][f], win_boxes[prev][f]
                for i, cc in enumerate(cols):
                    if cc in cb:
                        for j, pc in enumerate(pcols):
                            if pc in pb:
                                C[i, j] += iou(cb[cc], pb[pc])
            if len(cols) and len(pcols):
                for i, c in enumerate(cols):
                    j = int(np.argmax(C[i]))
                    if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                        maps[wi][c] = maps[prev][pcols[j]]
        for c in cols:
            if c not in maps[wi]:
                maps[wi][c] = nid; nid += 1
    acc = {}
    proto = None
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                proto = proto or b
                key = (gf, maps[wi][c])
                if key in acc:
                    x0 = min(acc[key][0], b.x); y0 = min(acc[key][1], b.y)
                    x1 = max(acc[key][0] + acc[key][2], b.x + b.w)
                    y1 = max(acc[key][1] + acc[key][3], b.y + b.h)
                    acc[key] = (x0, y0, x1 - x0, y1 - y0)
                else:
                    acc[key] = (b.x, b.y, b.w, b.h)
    return [type(proto)(gf, g, x, y, w, h, 1.0, -1, 1.0) for (gf, g), (x, y, w, h) in acc.items()]


def assemble_oracle():
    fresh, fid, out = {}, 100000, []
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                g = oracle[(wi, c)]
                if g is None:
                    if (wi, c) not in fresh:
                        fresh[(wi, c)] = fid; fid += 1
                    g = fresh[(wi, c)]
                out.append(type(b)(gf, g, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis))
    return out


out = {"video": VID, "chained": True, "n_windows": len(wins), "total_frames": total,
       "per_window_idf1": per_win, "in_window_mean": round(float(np.mean(per_win)), 3),
       "independent_baseline": {"raw_color": 0.196, "stitched_v3": 0.466, "oracle": 0.698,
                                "in_window_mean": 0.661},
       "modes": {}}
for name, pred in [("raw_color", assemble_raw()), ("stitched_v3", stitch_many()),
                   ("oracle", assemble_oracle())]:
    m5 = mot_metrics.evaluate(gt_all, pred, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt_all, pred, iou_threshold=0.3)
    out["modes"][name] = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
                              idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
                              pred_tracks=len({b.track_id for b in pred}))
    log(f"{name:<12} IDF1@0.5 {out['modes'][name]['idf1_05']:<6} IDF1@0.3 {out['modes'][name]['idf1_03']:<6} "
        f"IDSW {out['modes'][name]['idsw']:<5} MOTA {out['modes'][name]['mota_05']:<7} "
        f"tracks {out['modes'][name]['pred_tracks']}")

with open(f"{RES}/metrics_chained.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_chained.json")
