# Databricks notebook source
# ORACLE-STITCH headroom diagnostic (CPU, reuses the 17 generated windows -- no GPU).
# Question: of the in-window -> stitched IDF1 gap, how much is closable by ANY post-hoc stitcher?
#   (a) per-window IDF1 for each of the 17 windows (the honest in-window ceiling for THIS gen set);
#   (b) ORACLE stitch: assign each window-tracklet (window, colour) to its best-supported GT track
#       (summed per-frame IoU) -> global id = GT id. Upper bound of any post-hoc association;
#   (c) v1-stitch boundary diagnostics vs the oracle: link precision / link recall per boundary,
#       within-window split rate (GT covered by 2+ tracklets = flicker), merge rate.
# Decision rule (pre-registered): oracle ~= in-window mean -> gap closable, build graph stitcher;
# oracle ~= current stitched -> post-hoc maxed, the lever is generation-side overlap conditioning.

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
    VID = dbutils.widgets.get("vid") or "dancetrack0065"      # noqa: F821
    SAMP = dbutils.widgets.get("samp") or f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples"  # noqa: F821,E501
except Exception:
    VID, SAMP = "dancetrack0065", f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples"
RES = f"{VOL}/results/ltx_fullseq_{VID}"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, OVERLAP_LINK = 48, 150, 0.5
ORACLE_MIN_CREDIT = 0.5        # summed per-frame IoU a tracklet needs to be oracle-assigned
palette = build(K)

meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{VID}/windows.json"))
wins = sorted(meta["windows"], key=lambda w: w["start"])
total = meta["total_frames"]
log(f"{VID}: {len(wins)} windows, {total} frames")


def right_half(d):
    out = []
    for p in sorted(glob.glob(f"{d}/*.png")):
        f = np.asarray(imageio.imread(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        out.append(f)
    return out


def iou(a, b):
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    iX, iY = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a.w * a.h + b.w * b.h - inter
    return inter / uni if uni > 0 else 0.0


# decode all windows -> win_boxes[wi] = {global_frame: {colour: box}}
win_boxes, win_colours = [], []
for wi, w in enumerate(wins):
    frames = right_half(f"{SAMP}/step_001500_{wi+1}_png")
    if not frames:
        log(f"window {wi}: NO frames -> abort"); raise SystemExit(1)
    perf = {}
    for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
        perf.setdefault(w["start"] + (b.frame - 1), {})[b.track_id] = b
    win_boxes.append(perf)
    win_colours.append(sorted({c for d in perf.values() for c in d}))

seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w0, h0 = seq.image_size
sx, sy = 512.0 / w0, 512.0 / h0
gt_by_frame = {}
for f in range(1, total + 1):
    gt_by_frame[f] = [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy,
                              b.conf, b.cls, b.vis) for b in seq.boxes_at(f)]
gt_all = [g for gs in gt_by_frame.values() for g in gs]

# (a) per-window IDF1 (in-window ceiling for this generation set)
per_win = []
for wi, w in enumerate(wins):
    pred = [type(b)(gf, c, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis)
            for gf, d in win_boxes[wi].items() for c, b in d.items()]
    gtw = [g for f in range(w["start"], w["end"] + 1) for g in gt_by_frame[f]]
    m = mot_metrics.evaluate(gtw, pred, iou_threshold=0.5)
    per_win.append(round(m["idf1"], 3))
log(f"(a) per-window IDF1@0.5: mean {np.mean(per_win):.3f} min {min(per_win)} max {max(per_win)}")
log(f"    all: {per_win}")

# (b) oracle assignment: (wi, colour) -> best-supported GT id (or None)
oracle, splits, merges = {}, 0, 0
for wi, w in enumerate(wins):
    credit = {}
    for gf, d in win_boxes[wi].items():
        for c, b in d.items():
            for g in gt_by_frame.get(gf, []):
                v = iou(b, g)
                if v > 0:
                    credit.setdefault(c, Counter())[g.track_id] += v
    per_gt = Counter()
    for c in win_colours[wi]:
        best = credit.get(c, Counter()).most_common(1)
        if best and best[0][1] > ORACLE_MIN_CREDIT:
            oracle[(wi, c)] = best[0][0]
            per_gt[best[0][0]] += 1
        else:
            oracle[(wi, c)] = None
    splits += sum(n - 1 for n in per_gt.values() if n > 1)
n_assigned = sum(1 for v in oracle.values() if v is not None)
log(f"(b) oracle: {n_assigned}/{len(oracle)} tracklets assigned; within-window SPLITS (flicker) total {splits}")

# ownership split at overlap midpoints
own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]

# oracle full-seq eval
fresh, fid = {}, 100000
pred_oracle = []
for wi in range(len(wins)):
    for gf in range(own_start[wi], own_end[wi] + 1):
        for c, b in win_boxes[wi].get(gf, {}).items():
            g = oracle[(wi, c)]
            if g is None:
                if (wi, c) not in fresh:
                    fresh[(wi, c)] = fid; fid += 1
                g = fresh[(wi, c)]
            pred_oracle.append(type(b)(gf, g, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis))
mo5 = mot_metrics.evaluate(gt_all, pred_oracle, iou_threshold=0.5)
mo3 = mot_metrics.evaluate(gt_all, pred_oracle, iou_threshold=0.3)
log(f"(b) ORACLE full-seq IDF1@0.5 {mo5['idf1']:.3f} IDF1@0.3 {mo3['idf1']:.3f} IDSW {mo5['num_switches']}")


# (c) v1 stitch (full-overlap summed IoU, adjacent) + boundary diagnostics vs oracle
def stitch_v1():
    maps = [dict() for _ in wins]
    links = [[] for _ in wins]
    nid = 0
    for wi in range(len(wins)):
        cols = win_colours[wi]
        if wi > 0:
            prev = wi - 1
            ov = [f for f in range(wins[wi]["start"], wins[prev]["end"] + 1)
                  if f in win_boxes[wi] and f in win_boxes[prev]]
            pcols = win_colours[prev]
            if ov and cols and pcols:
                C = np.zeros((len(cols), len(pcols)))
                for f in ov:
                    cb, pb = win_boxes[wi][f], win_boxes[prev][f]
                    for i, cc in enumerate(cols):
                        if cc in cb:
                            for j, pc in enumerate(pcols):
                                if pc in pb:
                                    C[i, j] += iou(cb[cc], pb[pc])
                for i, j in zip(*linear_sum_assignment(-C)):
                    if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                        maps[wi][cols[i]] = maps[prev][pcols[j]]
                        links[wi].append((pcols[j], cols[i]))
        for c in cols:
            if c not in maps[wi]:
                maps[wi][c] = nid; nid += 1
    return maps, links


maps_v1, links = stitch_v1()
pred_v1 = [type(b)(gf, maps_v1[wi][c], b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis)
           for wi in range(len(wins)) for gf in range(own_start[wi], own_end[wi] + 1)
           for c, b in win_boxes[wi].get(gf, {}).items()]
mv5 = mot_metrics.evaluate(gt_all, pred_v1, iou_threshold=0.5)
log(f"(c) v1 stitched IDF1@0.5 {mv5['idf1']:.3f} (consistency check vs 0.413)")

correct = wrong = amb = carried = possible = 0
for wi in range(1, len(wins)):
    for (p, c) in links[wi]:
        gp, gc = oracle[(wi - 1, p)], oracle[(wi, c)]
        if gp is None or gc is None:
            amb += 1
        elif gp == gc:
            correct += 1
        else:
            wrong += 1
    prev_gts = {oracle[(wi - 1, p)] for p in win_colours[wi - 1]} - {None}
    cur_gts = {oracle[(wi, c)] for c in win_colours[wi]} - {None}
    for g in prev_gts & cur_gts:
        possible += 1
        if any(oracle[(wi - 1, p)] == g and oracle[(wi, c)] == g for (p, c) in links[wi]):
            carried += 1
prec = correct / max(1, correct + wrong)
rec = carried / max(1, possible)
log(f"(c) boundary links: precision {prec:.3f} ({correct} ok / {wrong} wrong / {amb} ambiguous), "
    f"recall {rec:.3f} ({carried}/{possible} GT identities carried across boundaries)")

out = dict(video=VID, n_windows=len(wins), total_frames=total,
           per_window_idf1=per_win, in_window_mean=round(float(np.mean(per_win)), 3),
           oracle=dict(idf1_05=round(mo5["idf1"], 3), idf1_03=round(mo3["idf1"], 3),
                       idsw=int(mo5["num_switches"]), assigned=n_assigned,
                       total_tracklets=len(oracle), within_window_splits=splits,
                       min_credit=ORACLE_MIN_CREDIT),
           v1=dict(idf1_05=round(mv5["idf1"], 3), link_precision=round(prec, 3),
                   link_recall=round(rec, 3), links_correct=correct, links_wrong=wrong,
                   links_ambiguous=amb, carried=carried, possible=possible))
with open(f"{RES}/metrics_oracle.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_oracle.json")
