# Databricks notebook source
# Render aligned (RGB clip, ID-map clip) pairs as LOSSLESS rgb24 video + a metadata.jsonl for the
# Lightricks ltx-trainer IC-LoRA (v2v) — the native, positionally-aligned conditioning that
# replaces the abandoned hand-rolled cond_sanity (clean negative).
#
# ID-maps MUST stay lossless (working-agreement rule 2, README.md). The trainer's read_video uses PyAV
# `frame.to_ndarray(format="rgb24")`, so FFV1 + rgb24 (no chroma subsampling) round-trips the flat
# colours EXACTLY into the Video VAE (which we already validated round-trips ID-maps at IoU 0.95).
# NEVER write the ID-map clip as yuv420p / lossy mp4 — that smears the palette and breaks decode.
#
# Trainer dataset (process_dataset.py auto-detects columns):
#   reference_video -> Video VAE -> reference_latents/   (our RGB clip = the conditioning)
#   video           -> Video VAE -> latents/             (our ID-map clip = the target)
#   caption         -> Gemma     -> conditions/          (the fixed task prompt)
# Same frame count, sequence-stable colours (one colour per dancer across the clip).

import json
import os
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)


# build_clips -> palette imports skimage; render needs scipy. Install before the gentrack import.
_pip("imageio", "imageio-ffmpeg", "scikit-image", "scipy")
import imageio.v2 as imageio  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

from gentrack.data.build_clips import iter_clips  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.palette import build  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
TRAIN_ROOT = f"{VOL}/dancetrack/train1"
PSEUDO = f"{VOL}/pseudo_masks/dancetrack"
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"

LENGTH = 49          # 8k+1, = our window; 7 latent frames
SIZE = 512
K = 48
HOP = 49             # non-overlapping windows
FPS = 24
import glob  # noqa: E402


def _w(name, default):
    try:
        v = dbutils.widgets.get(name)  # noqa: F821
        return v if v else default
    except Exception:
        return default


# split: demo (3 train seqs, sanity) | train (all 40, SAM silhouettes) | val (25 official, GT
# filled-box — no SAM on val, hygiene). Per-video window cap keeps the total clip count tractable.
SPLIT = _w("split", "demo")
OUT_SUB = _w("out_sub", "phase1")
MAX_PER_SEQ = int(_w("max_per_seq", "0"))    # 0 = no per-seq cap
MAX_CLIPS = int(_w("max_clips", "12"))       # global safety cap
DATASET = _w("dataset", "dancetrack")        # dancetrack | sportsmot
VIDS = [v.strip() for v in _w("vids", "").split(",") if v.strip()]   # optional subset (val selection)
if DATASET != "dancetrack":
    PSEUDO = f"{VOL}/pseudo_masks/{DATASET}"


def _seq_dirs(split_dirs):
    out = []
    for d in split_dirs:
        out += sorted(p for p in glob.glob(f"{VOL}/{DATASET}/{d}/*") if os.path.isdir(f"{p}/img1"))
    if VIDS:
        out = [p for p in out if os.path.basename(p) in VIDS]
    return out


if SPLIT == "train":
    SEQ_DIRS = _seq_dirs(["train1", "train2"] if DATASET == "dancetrack" else ["train"])
    MASK, PROOT, INCL_OCC = "sam_pseudo", PSEUDO, True
elif SPLIT == "val":
    SEQ_DIRS = _seq_dirs(["val"])
    MASK, PROOT, INCL_OCC = "filled_box", None, False   # val has GT boxes, no SAM (hygiene)
else:  # demo
    SEQ_DIRS = [f"{TRAIN_ROOT}/{s}" for s in
                ("dancetrack0008", "dancetrack0001", "dancetrack0002")]
    MASK, PROOT, INCL_OCC = "sam_pseudo", PSEUDO, True

OUT = f"{VOL}/ltx_ds/{OUT_SUB}"
os.makedirs(f"{OUT}/rgb", exist_ok=True)
os.makedirs(f"{OUT}/idmap", exist_ok=True)
log(f"dataset={DATASET} split={SPLIT} | {len(SEQ_DIRS)} seqs | mask={MASK} | out={OUT} | "
    f"max_per_seq={MAX_PER_SEQ} max_clips={MAX_CLIPS}")


def write_lossless(path, frames):
    """FFV1 / rgb24 (truly lossless, no chroma subsampling) so flat-colour ID-maps survive."""
    w = imageio.get_writer(path, fps=FPS, codec="ffv1", macro_block_size=1,
                           format="FFMPEG", output_params=["-pix_fmt", "rgb24"])
    for fr in frames:
        w.append_data(fr)
    w.close()


palette = build(K)
rows, n = [], 0
for sd in SEQ_DIRS:
    if not os.path.isdir(sd) or (MAX_CLIPS and n >= MAX_CLIPS):
        continue
    seq = load_sequence(sd)
    per = 0
    for frames, idmaps, spec in iter_clips(
            [seq], length=LENGTH, size=SIZE, hop=HOP, K=K, palette=palette,
            mask_source=MASK, pseudo_root=PROOT, include_occlusion=INCL_OCC):
        if (MAX_PER_SEQ and per >= MAX_PER_SEQ) or (MAX_CLIPS and n >= MAX_CLIPS):
            break
        tag = f"{spec.seq_name}_{spec.frames[0]:06d}_{spec.kind}"
        rgb_p, idm_p = f"{OUT}/rgb/{tag}.mkv", f"{OUT}/idmap/{tag}.mkv"
        write_lossless(rgb_p, frames)
        write_lossless(idm_p, idmaps)
        rows.append({"video": idm_p, "reference_video": rgb_p, "caption": PROMPT})
        per += 1; n += 1
    log(f"{spec.seq_name if per else os.path.basename(sd)}: +{per} (total {n})")

with open(f"{OUT}/metadata.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
log(f"DONE — {n} clip pairs from {len(SEQ_DIRS)} seqs -> {OUT}/metadata.jsonl")
