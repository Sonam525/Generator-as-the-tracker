# Databricks notebook source
# Path-B full-sequence: render ALL overlapping 49f windows (RGB reference clips) for one or more
# val videos. Overlap lets the stitcher/prefix carry identity across boundaries. Per video writes
# rgb/{tag}.mkv per window + windows.json (tag -> global frame range) under
# {VOL}/ltx_ds/fullseq/{vid}{out_suffix}/. Skips videos whose windows.json already exists.

import glob
import json
import os
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy"], check=True)
import imageio.v2 as imageio  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.data.build_clips import render_clip  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.palette import build  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
try:
    VIDS = [v.strip() for v in (dbutils.widgets.get("vids") or "dancetrack0065").split(",") if v.strip()]  # noqa: F821,E501
    W = int(dbutils.widgets.get("win") or "49")             # noqa: F821
    STRIDE = int(dbutils.widgets.get("stride") or "40")     # noqa: F821
    SUFFIX = dbutils.widgets.get("out_suffix") or "_ov9"    # noqa: F821
    SIZE = int(dbutils.widgets.get("size") or "512")        # noqa: F821
    DATASET = dbutils.widgets.get("dataset") or "dancetrack"  # noqa: F821  dancetrack | sportsmot
except Exception:
    VIDS, W, STRIDE, SUFFIX = ["dancetrack0065"], 49, 40, "_ov9"
    SIZE, DATASET = 512, "dancetrack"
if len(VIDS) == 1 and VIDS[0].lower() == "all_val":
    VIDS = sorted(os.path.basename(p) for p in glob.glob(f"{VOL}/{DATASET}/val/*") if os.path.isdir(f"{p}/img1"))

assert (W - 1) % 8 == 0, f"window {W} must be 8k+1"
assert SIZE % 32 == 0, f"size {SIZE} must be divisible by 32"
K, FPS = 48, 24
palette = build(K)


def write_lossless(path, frames):
    wtr = imageio.get_writer(path, fps=FPS, codec="ffv1", macro_block_size=1,
                             format="FFMPEG", output_params=["-pix_fmt", "rgb24"])
    for fr in frames:
        wtr.append_data(fr)
    wtr.close()


for VID in VIDS:
    OUT = f"{VOL}/ltx_ds/fullseq/{VID}{SUFFIX}"
    if os.path.exists(f"{OUT}/windows.json"):
        log(f"{VID}: windows.json exists -> skip")
        continue
    seq = load_sequence(f"{VOL}/{DATASET}/val/{VID}")
    total = seq.num_frames
    os.makedirs(f"{OUT}/rgb", exist_ok=True)
    starts = list(range(1, total - W + 2, STRIDE))
    if not starts or starts[-1] != total - W + 1:
        starts.append(total - W + 1)
    windows = []
    for wi, s in enumerate(starts):
        frames = list(range(s, s + W))
        rgb, _ = render_clip(seq, frames, palette, SIZE, K=K, mask_source="filled_box")
        tag = f"{VID}_w{wi:02d}_f{s:06d}"
        write_lossless(f"{OUT}/rgb/{tag}.mkv", rgb)
        windows.append(dict(idx=wi, tag=tag, start=s, end=s + W - 1))
    meta = dict(video=VID, total_frames=total, window=W, stride=STRIDE,
                overlap=W - STRIDE, n_windows=len(windows), windows=windows,
                image_size=list(seq.image_size))
    with open(f"{OUT}/windows.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"{VID}: {len(windows)} windows (win {W}, stride {STRIDE}) over {total} frames -> {OUT}")

log(f"RENDER-DONE {len(VIDS)} videos")
