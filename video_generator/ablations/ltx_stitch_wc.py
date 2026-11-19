# Databricks notebook source
# Path-B: quantify the WITHIN-WINDOW consolidation lever (CPU, reuses generated windows -- no GPU).
# Diagnosis showed the 0.413 ceiling on 0065 is decode-level colour flicker (one painted silhouette
# snaps between adjacent palette bins across frames -> extra track-ids INSIDE a window: 147 for 5 GT).
# Consolidation = link a window's per-frame boxes across frames by IoU into stable per-window tracklets
# (denoise the flicker), give each tracklet ONE representative colour, THEN stitch tracklets across
# window boundaries with the same overlap-IoU logic. Reports raw_color / stitched_plain / stitched_wc
# so the consolidation delta over the pure-colour 0.413 baseline is explicit.

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
K, MIN_AREA = 48, 150
LINK_IOU = 0.3          # within-window frame-to-frame IoU to continue a tracklet
MAX_GAP = 3             # a tracklet survives this many missed frames inside the window
OVERLAP_LINK = 0.5      # summed overlap-IoU to link tracklets across a window boundary
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


def iou(a, b):
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    iX, iY = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a.w * a.h + b.w * b.h - inter
    return inter / uni if uni > 0 else 0.0


# decode -> per window: {global_frame: {colour: box}}
win_raw = []
for wi, w in enumerate(wins):
    frames = right_half(f"{SAMP}/step_001500_{wi+1}_png")
    if not frames:
        log(f"window {wi}: NO frames -> abort"); raise SystemExit(1)
    perf = {}
    for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
        perf.setdefault(w["start"] + (b.frame - 1), {})[b.track_id] = b
    win_raw.append(perf)


def consolidate(perf):
    """Link per-frame boxes across a window's frames by IoU into tracklets (flicker denoise).
    Returns {tracklet_id: {global_frame: box}} and the tracklet's representative colour."""
    tracklets = []   # {boxes:{f:box}, last_f, last_box, colours:Counter}
    for f in sorted(perf):
        dets = list(perf[f].items())                       # (colour, box)
        live = [i for i, t in enumerate(tracklets) if f - t["last_f"] <= MAX_GAP]
        matched_d, matched_t = set(), set()
        if live and dets:
            C = np.zeros((len(dets), len(live)))
            for i, (_, b) in enumerate(dets):
                for j, ti in enumerate(live):
                    C[i, j] = iou(b, tracklets[ti]["last_box"])
            for i, j in zip(*linear_sum_assignment(-C)):
                if C[i, j] > LINK_IOU:
                    ti = live[j]; col, b = dets[i]
                    tracklets[ti]["boxes"][f] = b; tracklets[ti]["last_f"] = f
                    tracklets[ti]["last_box"] = b; tracklets[ti]["colours"][col] += 1
                    matched_d.add(i); matched_t.add(ti)
        for i, (col, b) in enumerate(dets):
            if i not in matched_d:
                tracklets.append(dict(boxes={f: b}, last_f=f, last_box=b, colours=Counter({col: 1})))
    out, colour = {}, {}
    for tid, t in enumerate(tracklets):
        out[tid] = t["boxes"]
        colour[tid] = t["colours"].most_common(1)[0][0]
    return out, colour


# per-window consolidated tracklets
win_trk, win_col = [], []
for perf in win_raw:
    trk, col = consolidate(perf)
    win_trk.append(trk); win_col.append(col)
frag_before = np.mean([len({c for d in p.values() for c in d}) for p in win_raw])
frag_after = np.mean([len(t) for t in win_trk])
log(f"avg distinct units per window: raw colours {frag_before:.1f} -> consolidated tracklets {frag_after:.1f}")

# ownership split at overlap midpoints
own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]


def stitch_units(win_units):
    """win_units[wi] = {unit_id: {frame: box}}. Adjacent overlap-IoU match -> global ids."""
    maps = [dict() for _ in wins]
    nid = 0
    for wi in range(len(wins)):
        units = list(win_units[wi].keys())
        if wi > 0:
            prev = wi - 1
            povr = range(wins[wi]["start"], wins[prev]["end"] + 1)
            pu = list(win_units[prev].keys())
            C = np.zeros((len(units), len(pu)))
            for i, u in enumerate(units):
                for j, p in enumerate(pu):
                    C[i, j] = sum(iou(win_units[wi][u][f], win_units[prev][p][f])
                                  for f in povr if f in win_units[wi][u] and f in win_units[prev][p])
            if len(units) and len(pu):
                for i, j in zip(*linear_sum_assignment(-C)):
                    if C[i, j] > OVERLAP_LINK and pu[j] in maps[prev]:
                        maps[wi][units[i]] = maps[prev][pu[j]]
        for u in units:
            if u not in maps[wi]:
                maps[wi][u] = nid; nid += 1
    return maps


maps_wc = stitch_units(win_trk)

seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w0, h0 = seq.image_size
sx, sy = 512.0 / w0, 512.0 / h0
gt = [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis)
      for f in range(1, total + 1) for b in seq.boxes_at(f)]


def owned(gf, wi):
    return own_start[wi] <= gf <= own_end[wi]


def eval_mode(name, boxes):
    m5 = mot_metrics.evaluate(gt, boxes, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt, boxes, iou_threshold=0.3)
    r = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
             idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
             pred_tracks=len({b.track_id for b in boxes}))
    log(f"{name:<16} IDF1@0.5 {r['idf1_05']:<6} IDF1@0.3 {r['idf1_03']:<6} IDSW {r['idsw']:<5} "
        f"MOTA {r['mota_05']:<7} tracks {r['pred_tracks']}")
    return r


# raw_color (pure model colour) over owned frames
raw = [type(b)(gf, c, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis)
       for wi in range(len(wins)) for gf, d in win_raw[wi].items() if owned(gf, wi)
       for c, b in d.items()]
# stitched_wc: consolidated tracklets, global ids, owned frames
wc = []
for wi in range(len(wins)):
    for u, fb in win_trk[wi].items():
        g = maps_wc[wi][u]
        for gf, b in fb.items():
            if owned(gf, wi):
                wc.append(type(b)(gf, g, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis))

out = {"video": VID, "total_frames": total, "n_windows": len(wins),
       "frag_raw_per_win": round(float(frag_before), 1), "frag_consolidated_per_win": round(float(frag_after), 1),
       "gt_tracks": len({b.track_id for b in gt}),
       "params": dict(link_iou=LINK_IOU, max_gap=MAX_GAP, overlap_link=OVERLAP_LINK), "modes": {}}
out["modes"]["raw_color"] = eval_mode("raw_color", raw)
out["modes"]["stitched_wc"] = eval_mode("stitched_wc", wc)
with open(f"{RES}/metrics_wc.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_wc.json")
