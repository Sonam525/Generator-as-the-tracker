# Databricks notebook source
# Stitcher v4 = many-to-one chain (v3 winner, 0.466) + GLOBAL TRACK-MERGE pass (transitive healing).
# Oracle autopsy: remaining loss vs the 0.698 ceiling = permanent chain splits -- a missed boundary
# link forks a person into two parallel global tracks that never re-merge. v4 adds union-find over
# global tracks with two pinned evidence rules:
#   (a) co-occurring split halves: >=3 shared frames at mean IoU > 0.5 -> same person
#   (b) sequential healing: temporal gap <= 50 frames AND velocity-predicted box IoU > 0.3 -> re-join
# Params pinned a priori (CO_MIN_FRAMES=3, CO_IOU=0.5, GAP_MAX=50, GAP_IOU=0.3 -- GAP_IOU reuses the
# bank's REACQ_IOU). Modes reported: v1 / many / v4 so each increment is explicit.

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
    VID, SAMP = "dancetrack0065", f"{VOL}/checkpoints/ltx_v2v_ic_lora_fullseq/samples"
RES = f"{VOL}/results/ltx_fullseq_{VID}"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, OVERLAP_LINK = 48, 150, 0.5
CO_MIN_FRAMES, CO_IOU, GAP_MAX, GAP_IOU = 3, 0.5, 50, 0.3
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


def iou_t(a, b):
    """IoU of (x,y,w,h) tuples."""
    ix, iy = max(a[0], b[0]), max(a[1], b[1])
    iX = min(a[0] + a[2], b[0] + b[2]); iY = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a[2] * a[3] + b[2] * b[3] - inter
    return inter / uni if uni > 0 else 0.0


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
PROTO = next(iter(win_boxes[0][min(win_boxes[0])].values()))     # a Box to construct from

seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
w0, h0 = seq.image_size
sx, sy = 512.0 / w0, 512.0 / h0
gt_all = [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis)
          for f in range(1, total + 1) for b in seq.boxes_at(f)]

own_start = [wins[0]["start"]]
for k in range(1, len(wins)):
    own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]


def stitch(mode):
    """'v1' Hungarian 1-1 · 'many' argmax many-to-one."""
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
                                C[i, j] += iou_t(cb[cc].xywh if hasattr(cb[cc], 'xywh') else
                                                 (cb[cc].x, cb[cc].y, cb[cc].w, cb[cc].h),
                                                 (pb[pc].x, pb[pc].y, pb[pc].w, pb[pc].h))
            if len(cols) and len(pcols):
                if mode == "v1":
                    for i, j in zip(*linear_sum_assignment(-C)):
                        if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                            maps[wi][cols[i]] = maps[prev][pcols[j]]
                else:
                    for i, c in enumerate(cols):
                        j = int(np.argmax(C[i]))
                        if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                            maps[wi][c] = maps[prev][pcols[j]]
        for c in cols:
            if c not in maps[wi]:
                maps[wi][c] = nid; nid += 1
    return maps


def tracks_from(maps):
    """{gid: {frame: (x,y,w,h)}} over owned frames, same-frame union-merged."""
    tr = {}
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                g = maps[wi][c]
                cur = tr.setdefault(g, {}).get(gf)
                if cur:
                    x0 = min(cur[0], b.x); y0 = min(cur[1], b.y)
                    x1 = max(cur[0] + cur[2], b.x + b.w); y1 = max(cur[1] + cur[3], b.y + b.h)
                    tr[g][gf] = (x0, y0, x1 - x0, y1 - y0)
                else:
                    tr[g][gf] = (b.x, b.y, b.w, b.h)
    return tr


def global_merge(tr):
    """Union-find over global tracks: co-occurrence and gap-healing merges (pinned params)."""
    gids = sorted(tr)
    parent = {g: g for g in gids}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]; g = parent[g]
        return g

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    spans = {g: (min(tr[g]), max(tr[g])) for g in gids}
    n_co = n_gap = 0
    for i, a in enumerate(gids):
        for b in gids[i + 1:]:
            sa, ea = spans[a]; sb, eb = spans[b]
            if sa > sb:
                a2, b2 = b, a; sa, ea, sb, eb = spans[a2][0], spans[a2][1], spans[b2][0], spans[b2][1]
            else:
                a2, b2 = a, b
            shared = [f for f in tr[a2] if f in tr[b2]]
            if len(shared) >= CO_MIN_FRAMES:                      # (a) co-occurring split halves
                m = np.mean([iou_t(tr[a2][f], tr[b2][f]) for f in shared])
                if m > CO_IOU:
                    union(a2, b2); n_co += 1
            elif not shared and 0 < sb - ea <= GAP_MAX:           # (b) sequential healing
                fs = sorted(tr[a2])[-5:]
                (x1, y1, w1, h1) = tr[a2][fs[-1]]
                if len(fs) >= 2:
                    (x0, y0, w0, h0) = tr[a2][fs[0]]
                    dt = max(1, fs[-1] - fs[0])
                    vx = ((x1 + w1 / 2) - (x0 + w0 / 2)) / dt
                    vy = ((y1 + h1 / 2) - (y0 + h0 / 2)) / dt
                else:
                    vx = vy = 0.0
                df = sb - fs[-1]
                pred = (x1 + vx * df, y1 + vy * df, w1, h1)
                if iou_t(pred, tr[b2][sb]) > GAP_IOU:
                    union(a2, b2); n_gap += 1
    out = {}
    for g in gids:
        r = find(g)
        for f, box in tr[g].items():
            cur = out.setdefault(r, {}).get(f)
            if cur:
                x0 = min(cur[0], box[0]); y0 = min(cur[1], box[1])
                x1 = max(cur[0] + cur[2], box[0] + box[2]); y1 = max(cur[1] + cur[3], box[1] + box[3])
                out[r][f] = (x0, y0, x1 - x0, y1 - y0)
            else:
                out[r][f] = box
    log(f"global merge: {len(gids)} -> {len(out)} tracks ({n_co} co-occurrence, {n_gap} gap-healing)")
    return out


def eval_tracks(name, tr):
    pred = [type(PROTO)(f, g, x, y, w, h, 1.0, -1, 1.0)
            for g, fb in tr.items() for f, (x, y, w, h) in fb.items()]
    m5 = mot_metrics.evaluate(gt_all, pred, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt_all, pred, iou_threshold=0.3)
    r = dict(idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
             idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3), pred_tracks=len(tr))
    log(f"{name:<6} IDF1@0.5 {r['idf1_05']:<6} IDF1@0.3 {r['idf1_03']:<6} IDSW {r['idsw']:<5} "
        f"MOTA {r['mota_05']:<7} tracks {r['pred_tracks']}")
    return r


out = {"video": VID, "n_windows": len(wins), "total_frames": total,
       "oracle_ceiling_idf1_05": 0.698,
       "params": dict(overlap_link=OVERLAP_LINK, co_min_frames=CO_MIN_FRAMES, co_iou=CO_IOU,
                      gap_max=GAP_MAX, gap_iou=GAP_IOU),
       "modes": {}}
out["modes"]["v1"] = eval_tracks("v1", tracks_from(stitch("v1")))
tr_many = tracks_from(stitch("many"))
out["modes"]["many"] = eval_tracks("many", tr_many)
out["modes"]["v4"] = eval_tracks("v4", global_merge(tr_many))
with open(f"{RES}/metrics_v4.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_v4.json (oracle ceiling 0.698)")
