# Databricks notebook source
# Path-B length test: render ONE long clip (frames 1..L) of a val video -> rgb.mkv + GT idmap.mkv
# (lossless FFV1/rgb24, GT filled-box since val has no SAM). Feeds a single longer-than-49f
# generation to test whether the 49f-trained IC-LoRA holds up on longer sequences.

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
    VID = dbutils.widgets.get("vid") or "dancetrack0065"  # noqa: F821
    L = int(dbutils.widgets.get("length") or "121")       # noqa: F821
except Exception:
    VID, L = "dancetrack0065", 121

assert (L - 1) % 8 == 0, f"length {L} must be 8k+1 for LTX"
K, SIZE, FPS = 48, 512, 24
palette = build(K)
OUT = f"{VOL}/ltx_ds/longtest/{VID}_L{L}"
os.makedirs(f"{OUT}/rgb", exist_ok=True)
os.makedirs(f"{OUT}/idmap", exist_ok=True)


def write_lossless(path, frames):
    w = imageio.get_writer(path, fps=FPS, codec="ffv1", macro_block_size=1,
                           format="FFMPEG", output_params=["-pix_fmt", "rgb24"])
    for fr in frames:
        w.append_data(fr)
    w.close()


seq = load_sequence(f"{VOL}/dancetrack/val/{VID}")
frames = list(range(1, L + 1))
rgb, idmap = render_clip(seq, frames, palette, SIZE, K=K, mask_source="filled_box")
tag = f"{VID}_000001_L{L}"
write_lossless(f"{OUT}/rgb/{tag}.mkv", rgb)
write_lossless(f"{OUT}/idmap/{tag}.mkv", idmap)
log(f"DONE {VID} frames 1..{L} -> {OUT}/rgb/{tag}.mkv (+ idmap). gt tracks in clip: "
    f"{len({b.track_id for f in frames for b in seq.boxes_at(f)})}")
