# Databricks notebook source
# P3 LONG-GAP RE-ACQUISITION study on the full-video outputs (CPU, reuses both arms' cleaned
# windows -- no GPU). The founding question (README: what A/B/C compete on; image-arm banks
# all scored reacq=0). Mines GT gap events (track absent >= GAP_MIN frames, then reappears),
# then per arm asks: was the person tracked before the gap, re-found after (redetect), and with the
# SAME identity (reacq)? Bins: 9-24 short / 25-48 sub-window (video-attention hypothesis bin) /
# 49-150 cross-window (prefix-chain bin) / >150 very long. Exit-vs-occlusion tagged by border
# proximity of the last pre-gap GT box. Definitions pinned a priori; match IoU 0.3; pre/post
# identity = majority pred-id over the nearest 15 visible frames; redetect = >=3 matched frames.

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy"], check=True)
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
VIDS = ["dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
        "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
        "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
        "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
        "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097"]
ARMS = [("ltx_arm_a_chain", "raw_color"), ("ltx_arm_b_indep", "v3many")]
RES = f"{VOL}/results/ltx_reacq_study"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, MATCH_IOU, OVERLAP_LINK = 48, 150, 0.3, 0.5
GAP_MIN, CTX, REDETECT_MIN, BORDER_FRAC = 9, 15, 3, 0.02
GAP_BINS = [(9, 24, "short9-24"), (25, 48, "subwin25-48"), (49, 150, "xwin49-150"),
            (151, 10 ** 9, "long>150")]


def iou_t(a, b):
    ix, iy = max(a[0], b[0]), max(a[1], b[1])
    iX = min(a[0] + a[2], b[0] + b[2]); iY = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a[2] * a[3] + b[2] * b[3] - inter
    return inter / uni if uni > 0 else 0.0


def assemble_arm(arm, link, vid, meta):
    """Decode one video's cleaned windows -> {frame: [(gid,(x,y,w,h))]} (512 space, owned frames)."""
    palette = build(K)
    wins = sorted(meta["windows"], key=lambda w: w["idx"])
    win_boxes, win_colours = [], []
    for wi, w in enumerate(wins):
        frames = [np.asarray(imageio.imread(p))[:, :, :3]
                  for p in sorted(glob.glob(f"{VOL}/checkpoints/{arm}/{vid}/cleaned/w{wi:02d}/*.png"))]
        perf = {}
        for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
            perf.setdefault(w["start"] + (b.frame - 1), {})[b.track_id] = b
        win_boxes.append(perf)
        win_colours.append(sorted({c for d in perf.values() for c in d}))
    own_start = [wins[0]["start"]]
    for k in range(1, len(wins)):
        own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
    own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]

    if link == "v3many":
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
                    for i, c in enumerate(cols):
                        j = int(np.argmax(C[i]))
                        if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                            maps[wi][c] = maps[prev][pcols[j]]
            for c in cols:
                if c not in maps[wi]:
                    maps[wi][c] = nid; nid += 1
        gid_of = lambda wi, c: maps[wi][c]  # noqa: E731
    else:
        gid_of = lambda wi, c: c            # noqa: E731

    out = {}
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                out.setdefault(gf, []).append((gid_of(wi, c), (b.x, b.y, b.w, b.h)))
    return out


def process_video(vid):
    meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{vid}_ov9/windows.json"))
    total = meta["total_frames"]
    seq = load_sequence(f"{VOL}/dancetrack/val/{vid}")
    w0, h0 = seq.image_size
    sx, sy = 512.0 / w0, 512.0 / h0
    # GT per track: frame -> box (512 space)
    gt_tracks = {}
    for f in range(1, total + 1):
        for b in seq.boxes_at(f):
            gt_tracks.setdefault(b.track_id, {})[f] = (b.x * sx, b.y * sy, b.w * sx, b.h * sy)
    # mine gap events
    events = []
    for tid, fb in gt_tracks.items():
        fs = sorted(fb)
        for i in range(1, len(fs)):
            gap = fs[i] - fs[i - 1] - 1
            if gap >= GAP_MIN:
                lb = fb[fs[i - 1]]
                edge = (lb[0] <= BORDER_FRAC * 512 or lb[1] <= BORDER_FRAC * 512 or
                        lb[0] + lb[2] >= (1 - BORDER_FRAC) * 512 or lb[1] + lb[3] >= (1 - BORDER_FRAC) * 512)
                events.append(dict(tid=tid, last=fs[i - 1], back=fs[i], gap=gap,
                                   kind="exit" if edge else "occlusion"))
    if not events:
        return dict(vid=vid, events=[])

    res_events = []
    for arm, link in ARMS:
        pred = assemble_arm(arm, link, vid, meta)
        for ev in events:
            fb = gt_tracks[ev["tid"]]
            pre_fs = [f for f in sorted(fb) if f <= ev["last"]][-CTX:]
            post_fs = [f for f in sorted(fb) if f >= ev["back"]][:CTX]

            def match_ids(frames_list):
                ids = []
                for f in frames_list:
                    g = fb[f]
                    cands = [(gid, iou_t(p, g)) for gid, p in pred.get(f, [])]
                    cands = [c for c in cands if c[1] >= MATCH_IOU]
                    if cands:
                        ids.append(max(cands, key=lambda c: c[1])[0])
                return ids

            pre_ids, post_ids = match_ids(pre_fs), match_ids(post_fs)
            tracked_before = len(pre_ids) >= REDETECT_MIN
            redetect = len(post_ids) >= REDETECT_MIN
            reacq = False
            if tracked_before and redetect:
                reacq = Counter(pre_ids).most_common(1)[0][0] == Counter(post_ids).most_common(1)[0][0]
            res_events.append(dict(vid=vid, arm=arm, **{k: ev[k] for k in ("tid", "gap", "kind")},
                                   tracked_before=tracked_before, redetect=redetect, reacq=reacq))
    return dict(vid=vid, events=res_events)


log(f"mining gaps >= {GAP_MIN} frames over {len(VIDS)} videos; arms: {[a for a, _ in ARMS]}")
with Pool(4) as pool:
    results = pool.map(process_video, VIDS)

all_events = [e for r in results for e in r["events"]]
log(f"total arm-events: {len(all_events)} ({len(all_events) // max(1, len(ARMS))} gap events x {len(ARMS)} arms)")

summary = {}
for arm, _ in ARMS:
    evs = [e for e in all_events if e["arm"] == arm]
    s = {}
    for lo, hi, lbl in GAP_BINS + [(GAP_MIN, 10 ** 9, "ALL")]:
        sel = [e for e in evs if lo <= e["gap"] <= hi]
        den = [e for e in sel if e["tracked_before"] and e["redetect"]]
        s[lbl] = dict(events=len(sel),
                      tracked_before=sum(e["tracked_before"] for e in sel),
                      redetect=sum(e["redetect"] for e in sel),
                      reacq=sum(e["reacq"] for e in sel),
                      reacq_rate_conditional=round(sum(e["reacq"] for e in den) / len(den), 3) if den else None,
                      cond_n=len(den))
    for kind in ("occlusion", "exit"):
        sel = [e for e in evs if e["kind"] == kind]
        den = [e for e in sel if e["tracked_before"] and e["redetect"]]
        s[f"kind_{kind}"] = dict(events=len(sel), reacq=sum(e["reacq"] for e in sel),
                                 reacq_rate_conditional=round(sum(e["reacq"] for e in den) / len(den), 3) if den else None,
                                 cond_n=len(den))
    summary[arm] = s
    log(f"--- {arm} ---")
    for lbl, d in s.items():
        log(f"  {lbl:<14} n={d['events']:<4} " +
            (f"tracked_pre={d.get('tracked_before')} redetect={d.get('redetect')} " if "tracked_before" in d else "") +
            f"reacq={d['reacq']} cond_rate={d['reacq_rate_conditional']} (cond_n={d['cond_n']})")

with open(f"{RES}/reacq_summary.json", "w") as f:
    json.dump(dict(params=dict(gap_min=GAP_MIN, ctx=CTX, match_iou=MATCH_IOU,
                               redetect_min=REDETECT_MIN, border_frac=BORDER_FRAC),
                   summary=summary), f, indent=2)
with open(f"{RES}/reacq_events.json", "w") as f:
    json.dump(all_events, f)
log(f"DONE -> {RES}/reacq_summary.json (+ per-event dump)")
