# Databricks notebook source
# Render RGB-only reference windows for the DanceTrack TEST split (no GT exists -- submission run).
# Same conventions as val: 49f windows, stride 40 (overlap 9), 512x512 via the training
# _resize_frame (BILINEAR square) so the reference distribution matches training exactly.
# Discovers test videos by glob; writes rgb/{tag}.mkv + windows.json per video; skips done videos.

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
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
try:
    W = int(dbutils.widgets.get("win") or "49")             # noqa: F821
    STRIDE = int(dbutils.widgets.get("stride") or "40")     # noqa: F821
    SIZE = int(dbutils.widgets.get("size") or "512")        # noqa: F821
    SHARD = dbutils.widgets.get("shard") or "0/1"           # noqa: F821  "i/n" video sharding
except Exception:
    W, STRIDE, SIZE, SHARD = 49, 40, 512, "0/1"
assert (W - 1) % 8 == 0 and SIZE % 32 == 0
FPS = 24
si, sn = (int(x) for x in SHARD.split("/"))

# test extracts as test1/ + test2/ (like train1/train2) -> union
seqdirs = sorted(glob.glob(f"{VOL}/dancetrack/test1/dancetrack*") +
                 glob.glob(f"{VOL}/dancetrack/test2/dancetrack*"))
seqdirs = [p for i, p in enumerate(seqdirs) if i % sn == si]
log(f"test videos in shard {SHARD}: {len(seqdirs)}")


def write_lossless(path, frames):
    wtr = imageio.get_writer(path, fps=FPS, codec="ffv1", macro_block_size=1,
                             format="FFMPEG", output_params=["-pix_fmt", "rgb24"])
    for fr in frames:
        wtr.append_data(fr)
    wtr.close()


for seqdir in seqdirs:
    VID = os.path.basename(seqdir)
    OUT = f"{VOL}/ltx_ds/fullseq/{VID}_ov9"
    if os.path.exists(f"{OUT}/windows.json"):
        log(f"{VID}: done, skip")
        continue
    seq = load_sequence(seqdir)                           # tolerates missing gt.txt
    total = seq.num_frames
    os.makedirs(f"{OUT}/rgb", exist_ok=True)
    starts = list(range(1, total - W + 2, STRIDE))
    if not starts or starts[-1] != total - W + 1:
        starts.append(total - W + 1)
    windows = []
    for wi, s in enumerate(starts):
        rgb = [_resize_frame(seq.frame_path(f), SIZE) for f in range(s, s + W)]
        tag = f"{VID}_w{wi:02d}_f{s:06d}"
        write_lossless(f"{OUT}/rgb/{tag}.mkv", rgb)
        windows.append(dict(idx=wi, tag=tag, start=s, end=s + W - 1))
    meta = dict(video=VID, split="test", total_frames=total, window=W, stride=STRIDE,
                overlap=W - STRIDE, n_windows=len(windows), windows=windows,
                image_size=list(seq.image_size))
    with open(f"{OUT}/windows.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"{VID}: {len(windows)} windows over {total} frames")

log(f"RENDER-TEST-DONE shard {SHARD}: {len(seqdirs)} videos")
