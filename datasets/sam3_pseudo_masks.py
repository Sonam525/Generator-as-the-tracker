# Databricks notebook source
# Full SAM 3.1 pseudo-mask pass (Mode 1) over the DanceTrack TRAIN splits only (val
# hygiene, rule 5). One uint16 instance-map PNG per frame → volume; identity from GT.
# Resumable (skips frames already written). Writes running per-sequence + aggregate
# coverage stats so progress is pollable. Long job — launch detached.

import json
import os
import subprocess
import sys
import time
import traceback


def pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

pip("-U", "transformers", "accelerate", "safetensors", "sentencepiece", "pillow")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.pseudo_masks import Sam3Masker, generate_for_sequence  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
try:
    DATASET = dbutils.widgets.get("dataset") or "dancetrack"   # noqa: F821  dancetrack | sportsmot
except Exception:
    DATASET = "dancetrack"
OUT = f"{VOL}/pseudo_masks/{DATASET}"
RES = f"{VOL}/results/sam3_pseudo_masks" + ("" if DATASET == "dancetrack" else f"_{DATASET}")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# TRAIN split only, by construction (hygiene): DanceTrack train1/train2, SportsMOT train/.
TRAIN_DIRS = ["train1", "train2"] if DATASET == "dancetrack" else ["train"]

seqs = []
for d in TRAIN_DIRS:
    root = f"{VOL}/{DATASET}/{d}"
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if os.path.isdir(f"{root}/{name}/img1"):
                seqs.append((name, f"{root}/{name}"))
log(f"{DATASET}: {len(seqs)} train sequences")

out = {"sequences": {}, "n_seqs": len(seqs)}
try:
    masker = Sam3Masker(hf_token=HF_TOKEN)
    t_start = time.time()
    tot_frames = tot_boxes = tot_matched = 0
    for i, (name, path) in enumerate(seqs):
        seq = load_sequence(path)
        t0 = time.time()
        stats = generate_for_sequence(seq, masker, f"{OUT}/{name}", iou_thr=0.5, resume=True)
        stats["sec"] = round(time.time() - t0, 1)
        stats["match_rate"] = round(stats["matched"] / max(1, stats["boxes"]), 3)
        out["sequences"][name] = stats
        tot_frames += stats["frames"]
        tot_boxes += stats["boxes"]
        tot_matched += stats["matched"]
        out["aggregate"] = {
            "done_seqs": i + 1,
            "new_frames": tot_frames,
            "overall_match_rate": round(tot_matched / max(1, tot_boxes), 3),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
        }
        log(f"[{i+1}/{len(seqs)}] {name}: {stats}")
        with open(f"{RES}/pass.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
    out["status"] = "OK"
    log(f"PASS OK: {tot_frames} new frames, match {out['aggregate']['overall_match_rate']}")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-1500:]
    log("PASS FAILED:\n" + out["error"])

with open(f"{RES}/pass.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("SAM3-PSEUDO-MASKS-DONE -> " + RES)
