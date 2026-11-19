# Databricks notebook source
# Full-scale arm eval: 25 val videos, FULL sequences. For a given arm's published cleaned windows,
# per video: decode -> assemble full-video tracks (link_mode raw_color for Arm A, v3 many-to-one for
# Arm B) -> IDF1/IDSW/MOTA via motmetrics (512 space) -> MOTChallenge txt (original res). Then one
# TrackEval pass over all 25 -> HOTA/AssA/DetA/IDF1 (the leaderboard-comparable numbers).
# Videos are processed in a multiprocessing Pool (decode is the bottleneck).

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
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy", "motmetrics"], check=True)
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.eval import mot_metrics  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
try:
    ARM = dbutils.widgets.get("arm_dir") or "ltx_arm_a_chain"      # noqa: F821
    LINK = dbutils.widgets.get("link_mode") or "raw_color"         # noqa: F821  raw_color|v3many
    SUFFIX = dbutils.widgets.get("win_suffix") or "_ov9"           # noqa: F821
    NPROC = int(dbutils.widgets.get("nproc") or "4")               # noqa: F821
except Exception:
    ARM, LINK, SUFFIX, NPROC = "ltx_arm_a_chain", "raw_color", "_ov9", 4

VIDS = ["dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
        "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
        "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
        "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
        "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097"]
ARMROOT = f"{VOL}/checkpoints/{ARM}"
RES = f"{VOL}/results/{ARM}_eval"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA, OVERLAP_LINK = 48, 150, 0.5
TRACKER = f"{ARM}_{LINK}"
ROOT = "/local_disk0/te"
BENCH, SPLIT = "GENTRACK", "val"
GTD = f"{ROOT}/gt/mot_challenge/{BENCH}-{SPLIT}"
TRD = f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{TRACKER}/data"
os.makedirs(TRD, exist_ok=True)
os.makedirs(f"{ROOT}/gt/mot_challenge/seqmaps", exist_ok=True)


def iou(a, b):
    ix, iy = max(a.x, b.x), max(a.y, b.y)
    iX, iY = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a.w * a.h + b.w * b.h - inter
    return inter / uni if uni > 0 else 0.0


def process_video(vid):
    """Decode cleaned windows -> full-video tracks -> per-video metrics + MOT lines."""
    palette = build(K)
    meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{vid}{SUFFIX}/windows.json"))
    wins = sorted(meta["windows"], key=lambda w: w["idx"])
    total = meta["total_frames"]
    win_boxes, win_colours = [], []
    for wi, w in enumerate(wins):
        frames = [np.asarray(imageio.imread(p))[:, :, :3]
                  for p in sorted(glob.glob(f"{ARMROOT}/{vid}/cleaned/w{wi:02d}/*.png"))]
        if not frames:
            return dict(vid=vid, error=f"no cleaned frames w{wi:02d}")
        perf = {}
        for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
            perf.setdefault(w["start"] + (b.frame - 1), {})[b.track_id] = b
        win_boxes.append(perf)
        win_colours.append(sorted({c for d in perf.values() for c in d}))
    own_start = [wins[0]["start"]]
    for k in range(1, len(wins)):
        own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
    own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]

    if LINK == "v3many":
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
        gid_of = lambda wi, c: maps[wi][c]  # noqa: E731
    else:                                   # raw_color: the model's colour IS the global id
        gid_of = lambda wi, c: c            # noqa: E731

    acc = {}
    proto = None
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            for c, b in win_boxes[wi].get(gf, {}).items():
                proto = proto or b
                key = (gf, gid_of(wi, c))
                if key in acc:
                    x0 = min(acc[key][0], b.x); y0 = min(acc[key][1], b.y)
                    x1 = max(acc[key][0] + acc[key][2], b.x + b.w)
                    y1 = max(acc[key][1] + acc[key][3], b.y + b.h)
                    acc[key] = (x0, y0, x1 - x0, y1 - y0)
                else:
                    acc[key] = (b.x, b.y, b.w, b.h)
    pred = [type(proto)(gf, g, x, y, w, h, 1.0, -1, 1.0) for (gf, g), (x, y, w, h) in acc.items()]

    seq = load_sequence(f"{VOL}/dancetrack/val/{vid}")
    w0, h0 = seq.image_size
    sx, sy = 512.0 / w0, 512.0 / h0
    gt = [type(b)(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis)
          for f in range(1, total + 1) for b in seq.boxes_at(f)]
    m5 = mot_metrics.evaluate(gt, pred, iou_threshold=0.5)
    m3 = mot_metrics.evaluate(gt, pred, iou_threshold=0.3)
    # MOT lines at ORIGINAL resolution for TrackEval
    ux, uy = w0 / 512.0, h0 / 512.0
    lines = [f"{b.frame},{b.track_id},{b.x*ux:.2f},{b.y*uy:.2f},{b.w*ux:.2f},{b.h*uy:.2f},1,-1,-1,-1"
             for b in pred]
    gtl = [f"{f},{b.track_id},{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},1,1,1"
           for f in range(1, total + 1) for b in seq.boxes_at(f)]
    return dict(vid=vid, frames=total, idf1_05=round(m5["idf1"], 3), idf1_03=round(m3["idf1"], 3),
                idsw=int(m5["num_switches"]), mota_05=round(m5["mota"], 3),
                gt_tracks=len({b.track_id for b in gt}), pred_tracks=len({b.track_id for b in pred}),
                mot_lines=lines, gt_lines=gtl, imsize=(w0, h0))


log(f"arm={ARM} link={LINK} -> tracker '{TRACKER}'")
with Pool(NPROC) as pool:
    results = pool.map(process_video, VIDS)

per, seqmap = {}, ["name"]
agg = dict(idf1_05=[], idf1_03=[], idsw=[], mota_05=[])
for r in results:
    if "error" in r:
        log(f"{r['vid']}: ERROR {r['error']}"); continue
    vid = r["vid"]
    per[vid] = {k: r[k] for k in ("frames", "idf1_05", "idf1_03", "idsw", "mota_05",
                                  "gt_tracks", "pred_tracks")}
    for k in ("idf1_05", "idf1_03", "idsw", "mota_05"):
        agg[k].append(r[k])
    with open(f"{TRD}/{vid}.txt", "w") as f:
        f.write("\n".join(r["mot_lines"]) + "\n")
    os.makedirs(f"{GTD}/{vid}/gt", exist_ok=True)
    with open(f"{GTD}/{vid}/gt/gt.txt", "w") as f:
        f.write("\n".join(r["gt_lines"]) + "\n")
    w0, h0 = r["imsize"]
    with open(f"{GTD}/{vid}/seqinfo.ini", "w") as f:
        f.write(f"[Sequence]\nname={vid}\nimDir=img1\nframeRate=20\nseqLength={r['frames']}\n"
                f"imWidth={w0}\nimHeight={h0}\nimExt=.jpg\n")
    seqmap.append(vid)
    log(f"{vid}: IDF1@0.5 {r['idf1_05']} @0.3 {r['idf1_03']} IDSW {r['idsw']} "
        f"MOTA {r['mota_05']} ({r['pred_tracks']} pred / {r['gt_tracks']} gt tracks, {r['frames']}f)")
open(f"{ROOT}/gt/mot_challenge/seqmaps/{BENCH}-{SPLIT}.txt", "w").write("\n".join(seqmap) + "\n")

out = dict(arm=ARM, link_mode=LINK, n_videos=len(per),
           mean_idf1_05=round(float(np.mean(agg["idf1_05"])), 3),
           mean_idf1_03=round(float(np.mean(agg["idf1_03"])), 3),
           mean_mota_05=round(float(np.mean(agg["mota_05"])), 3),
           total_idsw=int(np.sum(agg["idsw"])), per_video=per)
log(f"=== FULL-VIDEO mean IDF1@0.5 {out['mean_idf1_05']} | @0.3 {out['mean_idf1_03']} | "
    f"MOTA {out['mean_mota_05']} | IDSW {out['total_idsw']} over {out['n_videos']} videos ===")

# TrackEval -> HOTA
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)  # noqa: E731
sh("git clone --depth 1 https://github.com/JonathonLuiten/TrackEval /local_disk0/TrackEval")
sh(r"""find /local_disk0/TrackEval/trackeval -name '*.py' -exec sed -i -E 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} +""")
r = sh(f"python /local_disk0/TrackEval/scripts/run_mot_challenge.py "
       f"--GT_FOLDER {ROOT}/gt/mot_challenge --TRACKERS_FOLDER {ROOT}/trackers/mot_challenge "
       f"--BENCHMARK {BENCH} --SPLIT_TO_EVAL {SPLIT} --TRACKERS_TO_EVAL {TRACKER} "
       f"--METRICS HOTA CLEAR Identity --DO_PREPROC False --USE_PARALLEL True --NUM_PARALLEL_CORES {NPROC} "
       f"--PRINT_CONFIG False --OUTPUT_SUMMARY True --OUTPUT_DETAILED False --PLOT_CURVES False")
with open(f"{RES}/trackeval_stdout.txt", "w") as f:
    f.write(r.stdout + "\n---STDERR---\n" + r.stderr)
summ = glob.glob(f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{TRACKER}/*summary.txt")
if summ:
    lines = [ln.split() for ln in open(summ[0]).read().splitlines() if ln.strip()]
    if len(lines) >= 2:
        d = dict(zip(lines[0], lines[-1]))
        for k in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW"):
            if k in d:
                out[f"trackeval_{k}"] = round(float(d[k]), 2)
    log(f"TrackEval: HOTA {out.get('trackeval_HOTA')} DetA {out.get('trackeval_DetA')} "
        f"AssA {out.get('trackeval_AssA')} IDF1 {out.get('trackeval_IDF1')}")
else:
    log("TrackEval summary missing — see trackeval_stdout.txt")

with open(f"{RES}/metrics_{LINK}.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"DONE -> {RES}/metrics_{LINK}.json")
