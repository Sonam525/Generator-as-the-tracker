# Databricks notebook source
# Zero-shot perception probe: does the DanceTrack-trained conditioned-P0 (i2v + SAM
# silhouettes) transfer to MOT17 (street pedestrians, varied scale) WITHOUT retraining?
# Run it on a few MOT17 val-half frames, decode -> boxes, measure detection recall vs GT
# @IoU {0.5,0.3,0.1}, save gen images. Decides whether MOT17 needs its own perception
# fine-tune before the re-id eval. GPU only, no API.

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

pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "transformers",
    "peft", "safetensors", "sentencepiece", "motmetrics")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gentrack.data.datasets import mot17_splits  # noqa: E402
from gentrack.data.mot_format import Box  # noqa: E402
from gentrack.eval.mot_metrics import detection_recall  # noqa: E402
from gentrack.models.cosmos import CosmosNanoAdapter  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
LORA = f"{VOL}/checkpoints/p0_stage1_sam/lora_best"   # DanceTrack-trained conditioned perceiver
RES = f"{VOL}/results/mot17_perceive_probe"
os.makedirs(RES, exist_ok=True)
SIZE = 1024
SEQS = ["MOT17-02", "MOT17-09", "MOT17-05"]   # 1080p, 1080p, 480p — scale variety
FPV = 3                                        # val-half frames per sequence
palette = build(48)
adapter = CosmosNanoAdapter(f"{VOL}/models/cosmos3_nano", LORA, size=SIZE)

_, val = mot17_splits(f"{VOL}/MOT17", "train")
val_by = {s.name.split("_")[0]: s for s in val}   # strip the _fX-Y suffix from frame_half

out = {}
for sname in SEQS:
    try:
        seq = val_by.get(sname)
        if seq is None:
            out[sname] = {"error": "not found"}
            continue
        w, h = seq.image_size
        sx, sy = SIZE / w, SIZE / h
        fs = np.linspace(1, seq.num_frames, FPV + 2, dtype=int)[1:-1]
        recalls = []
        for j, fnum in enumerate(fs):
            frame = _resize_frame(seq.frame_path(int(fnum)), SIZE)
            gen = adapter.generate(frame)
            pred = idmaps_to_boxes([gen], palette=palette, min_area=200)
            gt = [Box(1, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy) for b in seq.boxes_at(int(fnum))]
            pred1 = [Box(1, b.track_id, *b.xywh) for b in pred]
            rec = detection_recall(gt, pred1, iou_thrs=(0.5, 0.3, 0.1))
            recalls.append(rec)
            if j == 0:
                Image.fromarray(frame).save(f"{RES}/{sname}_frame.png")
                Image.fromarray(gen).save(f"{RES}/{sname}_gen.png")
        out[sname] = {
            "frames_tested": len(fs), "gt_per_frame": round(np.mean([r["n_gt"] for r in recalls]), 1),
            "recall@0.5": round(np.mean([r["recall@0.5"] for r in recalls]), 3),
            "recall@0.3": round(np.mean([r["recall@0.3"] for r in recalls]), 3),
            "recall@0.1": round(np.mean([r["recall@0.1"] for r in recalls]), 3),
        }
        log(f"{sname}: {out[sname]}")
    except Exception:
        out[sname] = {"error": traceback.format_exc()[-500:]}
        log(f"{sname} FAILED: {out[sname]['error']}")
    with open(f"{RES}/probe.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

log("MOT17-PERCEIVE-PROBE-DONE -> " + RES)
