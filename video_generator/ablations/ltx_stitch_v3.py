# Databricks notebook source
# Stitcher v3: MANY-TO-ONE boundary association (CPU, reuses generated windows -- no GPU).
# Oracle diagnosis (metrics_oracle.json): post-hoc ceiling 0.698 vs v1 0.413; v1 links are good
# (precision 0.905 / recall 0.887) -- the loss is Hungarian's 1-1 constraint vs 72 within-window
# flicker SPLITS (person = 2+ colour-tracklets): only one split-half can link, the rest spawn fresh
# ids. Fix = allow many current tracklets to join one previous track (what the oracle does).
# Modes: v1 (Hungarian 1-1, baseline) / v3_many (pure argmax many-to-one) / v3_hybrid (Hungarian
# primary + argmax merge pass for leftovers). Same-frame duplicates of a global id are UNION-merged
# (a spatially split person becomes one box). Link precision/recall vs oracle reported per mode.
# Note: design selected from the 0065 oracle diagnosis; the final method must validate on more videos.

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
    STEP = int(dbutils.widgets.get("step") or "1500")         # noqa: F821
    WSUB = dbutils.widgets.get("windows_sub") or VID          # noqa: F821
except Exception:
    VID, SAMP = "dancetrack0065", f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples"
    STEP, WSUB = 1500, "dancetrack0065"
RES = f"{VOL}/results/ltx_fullseq_{VID}"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, OVERLAP_LINK, ORACLE_MIN_CREDIT = 48, 150, 0.5, 0.5
palette = build(K)

meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{WSUB}/windows.json"))
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


win_boxes, win_colours = [], []
for wi, w in enumerate(wins):
    frames = right_half(f"{SAMP}/step_{STEP:06d}_{wi+1}_png")
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
gt_by_frame = {f: [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy,
                           b.conf, b.cls, b.vis) for b in seq.boxes_at(f)]
               for f in range(1, total + 1)}
gt_all = [g for gs in gt_by_frame.values() for g in gs]

# oracle assignment (diagnostics only -- never used in the stitchers)
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

own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]


def boundary_cost(wi):
    """C[i,j] = summed IoU over the FULL overlap between window wi colours and wi-1 colours."""
    prev = wi - 1
    ov = [f for f in range(wins[wi]["start"], wins[prev]["end"] + 1)
          if f in win_boxes[wi] and f in win_boxes[prev]]
    cols, pcols = win_colours[wi], win_colours[prev]
    C = np.zeros((len(cols), len(pcols)))
    for f in ov:
        cb, pb = win_boxes[wi][f], win_boxes[prev][f]
        for i, cc in enumerate(cols):
            if cc in cb:
                for j, pc in enumerate(pcols):
                    if pc in pb:
                        C[i, j] += iou(cb[cc], pb[pc])
    return C, cols, pcols


def stitch(mode):
    """mode: 'v1' Hungarian 1-1 Â· 'many' pure argmax many-to-one Â· 'hybrid' Hungarian + merge pass."""
    maps = [dict() for _ in wins]
    links = [[] for _ in wins]
    nid = 0
    for wi in range(len(wins)):
        cols = win_colours[wi]
        if wi > 0:
            C, cols, pcols = boundary_cost(wi)
            if len(cols) and len(pcols):
                if mode in ("v1", "hybrid"):
                    for i, j in zip(*linear_sum_assignment(-C)):
                        if C[i, j] > OVERLAP_LINK and pcols[j] in maps[wi - 1]:
                            maps[wi][cols[i]] = maps[wi - 1][pcols[j]]
                            links[wi].append((pcols[j], cols[i]))
                if mode in ("many", "hybrid"):
                    for i, c in enumerate(cols):           # leftovers (or all, for 'many') by argmax
                        if c in maps[wi]:
                            continue
                        j = int(np.argmax(C[i]))
                        if C[i, j] > OVERLAP_LINK and pcols[j] in maps[wi - 1]:
                            maps[wi][c] = maps[wi - 1][pcols[j]]
                            links[wi].append((pcols[j], c))
        for c in cols:
            if c not in maps[wi]:
                maps[wi][c] = nid; nid += 1
    return maps, links


def assemble(maps):
    """Owned frames; same-(frame, gid) duplicates union-merged into one box."""
    acc = {}
    proto = None
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                proto = proto or b
                g = maps[wi][c]
                key = (gf, g)
                if key in acc:
                    x0 = min(acc[key][0], b.x); y0 = min(acc[key][1], b.y)
                    x1 = max(acc[key][0] + acc[key][2], b.x + b.w)
                    y1 = max(acc[key][1] + acc[key][3], b.y + b.h)
                    acc[key] = (x0, y0, x1 - x0, y1 - y0)
                else:
                    acc[key] = (b.x, b.y, b.w, b.h)
    return [type(proto)(gf, g, x, y, w, h, 1.0, -1, 1.0) for (gf, g), (x, y, w, h) in acc.items()]


def link_stats(links):
    correct = wrong = amb = 0
    for wi in range(1, len(wins)):
        for (p, c) in links[wi]:
            gp, gc = oracle[(wi - 1, p)], oracle[(wi, c)]
            if gp is None or gc is None:
                amb += 1
            elif gp == gc:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, amb


out = {"video": VID, "n_windows": len(wins), "total_frames": total,
       "oracle_ceiling_idf1_05": 0.698, "modes": {}}
for mode in ["v1", "many", "hybrid"]:
    maps, links = stitch(mode)
    pred = assemble(maps)
    m5 = mot_metrics.evaluate(gt_all, pred, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt_all, pred, iou_threshold=0.3)
    c_, w_, a_ = link_stats(links)
    out["modes"][mode] = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
                              idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
                              pred_tracks=len({b.track_id for b in pred}),
                              links_correct=c_, links_wrong=w_, links_ambiguous=a_)
    log(f"{mode:<8} IDF1@0.5 {out['modes'][mode]['idf1_05']:<6} IDF1@0.3 {out['modes'][mode]['idf1_03']:<6} "
        f"IDSW {out['modes'][mode]['idsw']:<5} MOTA {out['modes'][mode]['mota_05']:<7} "
        f"tracks {out['modes'][mode]['pred_tracks']:<5} links ok/wrong/amb {c_}/{w_}/{a_}")

with open(f"{RES}/metrics_v3_{os.path.basename(SAMP.rstrip(chr(47)+chr(92)))}_{STEP}.json" if STEP != 1500 else f"{RES}/metrics_v3.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_v3.json (oracle ceiling 0.698)")
