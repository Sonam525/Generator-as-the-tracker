# Databricks notebook source
# Path A: HOTA via TrackEval on step-1500 (49-frame val windows). Format the step-1500 generations +
# val GT into MOTChallenge layout (pred scaled 512 -> original res to match GT), clone TrackEval, run
# HOTA/CLEAR/Identity. HOTA/AssA/DetA are the DanceTrack-leaderboard primaries.

import glob
import json
import os
import re
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sh(cmd, check=True):
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-6000:], flush=True)
    if r.stderr:
        print("STDERR:", r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")
    return r


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy"], check=True)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
SAMP = f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase2/samples"
VALROOT = f"{VOL}/dancetrack/val"
RES = f"{VOL}/results/ltx_hota_eval_step1500"
os.makedirs(RES, exist_ok=True)
STEP, K = 1500, 48
palette = build(K)
VIDS = ["dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
        "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
        "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
        "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
        "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097"]

BENCH, SPLIT, TRACKER = "GENTRACK", "val", "ltx_step1500"
ROOT = "/local_disk0/te"
GTD = f"{ROOT}/gt/mot_challenge/{BENCH}-{SPLIT}"
TRD = f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{TRACKER}/data"
os.makedirs(TRD, exist_ok=True)
os.makedirs(f"{ROOT}/gt/mot_challenge/seqmaps", exist_ok=True)


def right_half(d):
    out = []
    for p in sorted(glob.glob(f"{d}/*.png")):
        f = np.asarray(Image.open(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        out.append(f)
    return out


seqmap = ["name"]
for idx, vid in enumerate(VIDS, start=1):
    gen = right_half(f"{SAMP}/step_{STEP:06d}_{idx}_png")
    if not gen:
        log(f"{vid}: no gen, skip"); continue
    n = len(gen)
    pred = idmaps_to_boxes(gen, palette=palette, min_area=150)     # 512-space boxes, frame 1..n
    seq = load_sequence(f"{VALROOT}/{vid}")
    w, h = seq.image_size
    sx, sy = w / 512.0, h / 512.0                                   # 512 -> original res (match GT)
    with open(f"{TRD}/{vid}.txt", "w") as f:
        for b in pred:
            f.write(f"{b.frame},{b.track_id},{b.x*sx:.2f},{b.y*sy:.2f},{b.w*sx:.2f},{b.h*sy:.2f},1,-1,-1,-1\n")
    os.makedirs(f"{GTD}/{vid}/gt", exist_ok=True)
    with open(f"{GTD}/{vid}/gt/gt.txt", "w") as f:
        for fr in range(1, n + 1):
            for b in seq.boxes_at(fr):
                f.write(f"{fr},{b.track_id},{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},1,1,1\n")
    with open(f"{GTD}/{vid}/seqinfo.ini", "w") as f:
        f.write(f"[Sequence]\nname={vid}\nimDir=img1\nframeRate=20\nseqLength={n}\n"
                f"imWidth={w}\nimHeight={h}\nimExt=.jpg\n")
    seqmap.append(vid)
open(f"{ROOT}/gt/mot_challenge/seqmaps/{BENCH}-{SPLIT}.txt", "w").write("\n".join(seqmap) + "\n")
log(f"formatted {len(seqmap)-1} seqs into MOTChallenge layout")

sh("git clone --depth 1 https://github.com/JonathonLuiten/TrackEval /local_disk0/TrackEval", check=False)
# TrackEval uses np.float/np.int/np.bool aliases removed in NumPy 2.0 -> patch to builtins
sh(r"""find /local_disk0/TrackEval/trackeval -name '*.py' -exec sed -i -E 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} +""", check=False)
sh("pip install -q scipy", check=False)
cmd = (f"python /local_disk0/TrackEval/scripts/run_mot_challenge.py "
       f"--GT_FOLDER {ROOT}/gt/mot_challenge --TRACKERS_FOLDER {ROOT}/trackers/mot_challenge "
       f"--BENCHMARK {BENCH} --SPLIT_TO_EVAL {SPLIT} --TRACKERS_TO_EVAL {TRACKER} "
       f"--METRICS HOTA CLEAR Identity --DO_PREPROC False --USE_PARALLEL False "
       f"--NUM_PARALLEL_CORES 1 --PRINT_CONFIG False --OUTPUT_SUMMARY True --OUTPUT_DETAILED False --PLOT_CURVES False")
r = sh(cmd, check=False)
with open(f"{RES}/trackeval_stdout.txt", "w") as f:
    f.write(r.stdout + "\n---STDERR---\n" + r.stderr)

# parse the pedestrian_summary.txt (header line + values line, COMBINED)
out = {"checkpoint": f"step_{STEP}", "n_videos": len(seqmap) - 1, "note": "49-frame windows"}
summ = glob.glob(f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{TRACKER}/*summary.txt")
if summ:
    lines = [ln.split() for ln in open(summ[0]).read().splitlines() if ln.strip()]
    if len(lines) >= 2:
        hdr, val = lines[0], lines[-1]
        d = dict(zip(hdr, val))
        for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSw", "IDSW"):
            if k in d:
                out[k] = round(float(d[k]), 2)
    out["summary_raw"] = open(summ[0]).read()[:1500]
else:
    # fallback: scrape the stdout table for the HOTA/Identity COMBINED rows
    for m in re.finditer(r"(HOTA|IDF1|MOTA|DetA|AssA)\s*[:=]?\s*([0-9.]+)", r.stdout):
        out.setdefault(m.group(1), float(m.group(2)))
with open(f"{RES}/metrics.json", "w") as f:
    json.dump(out, f, indent=2)
log(f"HOTA-EVAL DONE -> {RES}/metrics.json | {[(k, out.get(k)) for k in ('HOTA','DetA','AssA','IDF1','MOTA')]}")
