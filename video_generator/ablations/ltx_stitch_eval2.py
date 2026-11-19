# Databricks notebook source
# Path-B stitcher v2 (CPU, reuses the already-generated windows -- no new GPU). Improves on v1
# (adjacent-window overlap matching only) with:
#   (1) within-window noise filter: drop colours present in < MIN_WIN_FRAMES frames of a window
#       (kills transient palette-snap speckle that spawns junk tracks),
#   (2) a global track BANK with motion-predicted RE-ACQUISITION: an unmatched colour is matched not
#       just to the previous window but to any recently-lost global track (last seen <= MAXAGE frames),
#       by extrapolating that track's box with its velocity + a raw-colour agreement bonus.
# Reports no_link / raw_color / stitched_v1 / stitched_v2 over the full sequence so the v2-vs-v1 delta
# is explicit. Params are pinned defaults (not tuned to this video); final method is validated on 25.

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
    MIN_WIN_FRAMES = int(dbutils.widgets.get("minwin") or "1")  # noqa: F821  (1 = filter off)
except Exception:
    VID, SAMP, MIN_WIN_FRAMES = "dancetrack0065", f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples", 1
RES = f"{VOL}/results/ltx_fullseq_{VID}"
os.makedirs(RES, exist_ok=True)

# pinned defaults
K, MIN_AREA = 48, 150       # MIN_WIN_FRAMES set above (widget; 1 = no within-window noise filter)
OVERLAP_LINK = 0.5          # overlap-IoU needed to link adjacent windows
REACQ_IOU = 0.3             # predicted-box IoU needed to re-acquire a lost track
MAXAGE = 150               # a lost track stays re-acquirable for this many frames
COLOR_BONUS = 0.15          # added to re-acq score when the raw palette colour matches
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


# decode each window -> {global_frame: {colour: Box}} + set of kept colours (noise-filtered)
win_boxes, win_colours = [], []
for wi, w in enumerate(wins):
    frames = right_half(f"{SAMP}/step_001500_{wi+1}_png")
    if not frames:
        log(f"window {wi}: NO frames -> abort"); raise SystemExit(1)
    boxes = idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA)
    perf, cnt = {}, {}
    for b in boxes:
        gf = w["start"] + (b.frame - 1)
        perf.setdefault(gf, {})[b.track_id] = b
        cnt[b.track_id] = cnt.get(b.track_id, 0) + 1
    kept = {c for c, k in cnt.items() if k >= MIN_WIN_FRAMES}          # (1) noise filter
    perf = {gf: {c: b for c, b in d.items() if c in kept} for gf, d in perf.items()}
    win_boxes.append(perf); win_colours.append(sorted(kept))

# ownership: split overlaps at the midpoint so each frame belongs to exactly one window
own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]


def predict_box(traj, target):
    """traj: list of (frame, Box) committed; extrapolate centre by recent velocity to `target`."""
    last_f, last_b = traj[-1]
    if len(traj) >= 2:
        f0, b0 = traj[max(0, len(traj) - 5)]
        dt = max(1, last_f - f0)
        vx = ((last_b.x + last_b.w / 2) - (b0.x + b0.w / 2)) / dt
        vy = ((last_b.y + last_b.h / 2) - (b0.y + b0.h / 2)) / dt
    else:
        vx = vy = 0.0
    df = target - last_f
    cx = last_b.x + last_b.w / 2 + vx * df
    cy = last_b.y + last_b.h / 2 + vy * df
    return type(last_b)(target, -1, cx - last_b.w / 2, cy - last_b.h / 2, last_b.w, last_b.h,
                        1.0, -1, 1.0)


def stitch(use_bank):
    """maps[wi][colour] = global_id. (1) adjacent match to the PREVIOUS window over the FULL overlap
    (reproduces v1); (2) if use_bank, re-acquire still-unmatched colours against genuinely LOST tracks
    (last seen before this window, within MAXAGE) by motion-predicted box; (3) else a new track."""
    maps = [dict() for _ in wins]
    banks = []   # global tracks: {traj:[(f,box)], last_seen, colours:set}

    def new_track(c):
        banks.append(dict(traj=[], last_seen=-10 ** 9, colours={c}))
        return len(banks) - 1

    for wi, w in enumerate(wins):
        cols = win_colours[wi]
        assigned = {}
        # (1) adjacent overlap match to previous window (full overlap, summed IoU) -> its global id
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
                ri, ci = linear_sum_assignment(-C)
                for i, j in zip(ri, ci):
                    if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                        assigned[cols[i]] = maps[prev][pcols[j]]
        # (2) re-acquire unmatched colours against lost tracks (motion-predicted)
        if use_bank:
            un = [c for c in cols if c not in assigned]
            first_f = {c: min(gf for gf, d in win_boxes[wi].items() if c in d) for c in un
                       if any(c in d for d in win_boxes[wi].values())}
            lost = [g for g in range(len(banks))
                    if banks[g]["last_seen"] < wins[wi]["start"]
                    and 0 <= wins[wi]["start"] - banks[g]["last_seen"] <= MAXAGE
                    and g not in assigned.values() and banks[g]["traj"]]
            if un and lost:
                C = np.full((len(un), len(lost)), -1e9)
                for i, c in enumerate(un):
                    if c not in first_f:
                        continue
                    tf = first_f[c]; cb = win_boxes[wi][tf][c]
                    for j, g in enumerate(lost):
                        pb = predict_box(banks[g]["traj"], tf)
                        C[i, j] = iou(pb, cb) + (COLOR_BONUS if c in banks[g]["colours"] else 0)
                ri, ci = linear_sum_assignment(-C)
                for i, j in zip(ri, ci):
                    if C[i, j] > REACQ_IOU:
                        assigned[un[i]] = lost[j]
        # (3) new tracks for the rest
        for c in cols:
            if c not in assigned:
                assigned[c] = new_track(c)
        maps[wi] = assigned
        for c, g in assigned.items():                             # commit owned frames + update bank
            banks[g]["colours"].add(c)
            for gf in range(own_start[wi], own_end[wi] + 1):
                b = win_boxes[wi].get(gf, {}).get(c)
                if b is not None:
                    banks[g]["traj"].append((gf, b))
                    banks[g]["last_seen"] = max(banks[g]["last_seen"], gf)
    return maps


maps_v1 = stitch(use_bank=False)
maps_v2 = stitch(use_bank=True)


def assemble(mode, maps=None):
    out = []
    for wi, w in enumerate(wins):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                if mode == "no_link":
                    gid = wi * 1000 + c
                elif mode == "raw_color":
                    gid = c
                else:
                    gid = maps[wi].get(c)
                    if gid is None:
                        continue
                out.append(type(b)(gf, gid, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis))
    return out


seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w0, h0 = seq.image_size
sx, sy = 512.0 / w0, 512.0 / h0
gt = [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis)
      for f in range(1, total + 1) for b in seq.boxes_at(f)]

out = {"video": VID, "total_frames": total, "n_windows": len(wins), "overlap": meta["overlap"],
       "gt_tracks": len({b.track_id for b in gt}),
       "params": dict(min_win_frames=MIN_WIN_FRAMES, overlap_link=OVERLAP_LINK,
                      reacq_iou=REACQ_IOU, maxage=MAXAGE, color_bonus=COLOR_BONUS),
       "modes": {}}
jobs = [("no_link", None), ("raw_color", None), ("stitched_v1", maps_v1), ("stitched_v2", maps_v2)]
for mode, mp in jobs:
    pred = assemble("stitched" if mode.startswith("stitched") else mode, mp)
    m5 = mot_metrics.evaluate(gt, pred, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt, pred, iou_threshold=0.3)
    out["modes"][mode] = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
                              idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
                              pred_tracks=len({b.track_id for b in pred}))
    log(f"{mode:<12} IDF1@0.5 {out['modes'][mode]['idf1_05']:<6} IDF1@0.3 {out['modes'][mode]['idf1_03']:<6} "
        f"IDSW {out['modes'][mode]['idsw']:<5} MOTA {out['modes'][mode]['mota_05']:<7} "
        f"tracks {out['modes'][mode]['pred_tracks']}")

with open(f"{RES}/metrics_v2.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_v2.json  (gt {out['gt_tracks']} tracks / {total} frames)")
