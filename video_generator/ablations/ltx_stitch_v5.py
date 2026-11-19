# Databricks notebook source
# Stitcher v5 = many-to-one chain + COLOUR-AFFINITY healing (the on-thesis merge).
# v4 result: geometric transitive merge fired ~never (+0.000) -- chain breaks happen exactly where
# geometry fails (full-overlap occlusion, fast nonlinear motion), so post-hoc GEOMETRY is exhausted
# at 0.466 vs the 0.698 oracle ceiling. The unused identity signal is the MODEL'S OWN COLOUR: the
# generator re-derives the same palette colour for the same person across windows (raw_colour alone
# scores 0.196 over 702f). v5 heals broken chains by colour: track b starting after track a ended,
# gap <= 50 frames, IDENTICAL dominant palette colour, both >= 5 frames long -> same person.
# Params pinned a priori. Reports v1 / many / v5 so colour-healing's increment is isolated.

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
GAP_MAX, MIN_LEN = 50, 5
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
PROTO = next(iter(win_boxes[0][min(win_boxes[0])].values()))

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
                                C[i, j] += iou_t((cb[cc].x, cb[cc].y, cb[cc].w, cb[cc].h),
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


def tracks_and_colours(maps):
    """{gid: {frame: box}} over owned frames (union-merged) + dominant palette colour per gid."""
    tr, colw = {}, {}
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                g = maps[wi][c]
                colw.setdefault(g, Counter())[c] += 1
                cur = tr.setdefault(g, {}).get(gf)
                if cur:
                    x0 = min(cur[0], b.x); y0 = min(cur[1], b.y)
                    x1 = max(cur[0] + cur[2], b.x + b.w); y1 = max(cur[1] + cur[3], b.y + b.h)
                    tr[g][gf] = (x0, y0, x1 - x0, y1 - y0)
                else:
                    tr[g][gf] = (b.x, b.y, b.w, b.h)
    dom = {g: colw[g].most_common(1)[0][0] for g in tr}
    return tr, dom


def colour_heal(tr, dom):
    """Union sequential same-dominant-colour tracks (gap <= GAP_MAX, both >= MIN_LEN frames)."""
    gids = [g for g in tr if len(tr[g]) >= MIN_LEN]
    parent = {g: g for g in tr}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]; g = parent[g]
        return g

    spans = {g: (min(tr[g]), max(tr[g])) for g in gids}
    pairs = []
    for i, a in enumerate(gids):
        for b in gids[i + 1:]:
            (sa, ea), (sb, eb) = spans[a], spans[b]
            if ea < sb and 0 < sb - ea <= GAP_MAX and dom[a] == dom[b]:
                pairs.append((sb - ea, a, b))
            elif eb < sa and 0 < sa - eb <= GAP_MAX and dom[a] == dom[b]:
                pairs.append((sa - eb, b, a))
    n = 0
    for gap, a, b in sorted(pairs):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # only heal if the merged spans stay sequential (no frame collision between components)
        fa = {f for g in tr if find(g) == ra for f in tr[g]}
        fb = {f for g in tr if find(g) == rb for f in tr[g]}
        if fa & fb:
            continue
        parent[rb] = ra; n += 1
    out = {}
    for g in tr:
        r = find(g)
        for f, box in tr[g].items():
            cur = out.setdefault(r, {}).get(f)
            if cur:
                x0 = min(cur[0], box[0]); y0 = min(cur[1], box[1])
                x1 = max(cur[0] + cur[2], box[0] + box[2]); y1 = max(cur[1] + cur[3], box[1] + box[3])
                out[r][f] = (x0, y0, x1 - x0, y1 - y0)
            else:
                out[r][f] = box
    log(f"colour healing: {len(tr)} -> {len(out)} tracks ({n} merges)")
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
       "params": dict(overlap_link=OVERLAP_LINK, gap_max=GAP_MAX, min_len=MIN_LEN), "modes": {}}
tr1, _ = tracks_and_colours(stitch("v1"))
out["modes"]["v1"] = eval_tracks("v1", tr1)
trm, dom = tracks_and_colours(stitch("many"))
out["modes"]["many"] = eval_tracks("many", trm)
out["modes"]["v5"] = eval_tracks("v5", colour_heal(trm, dom))
with open(f"{RES}/metrics_v5.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_v5.json (oracle ceiling 0.698)")
