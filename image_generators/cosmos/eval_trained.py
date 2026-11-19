# Databricks notebook source
# Identity eval for a trained Cosmos-Nano LoRA: chained tracking over the SAME clips
# P-A used (dt0004 f1-30, dt0004 gap-8 window, dt0005 gap-16 window) → IDF1/IDSW/MOTA +
# re-acquisition. Directly comparable to the P-A baseline. Writes results to the volume.

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

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
LORA = f"{VOL}/checkpoints/p1_stage2/lora_best"
RES = f"{VOL}/results/p1_identity_eval"
os.makedirs(RES, exist_ok=True)

from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.mot_format import Box  # noqa: E402
from gentrack.eval.chained_eval import chained_track  # noqa: E402
from gentrack.eval.mot_metrics import evaluate, reacquisition_check  # noqa: E402
from gentrack.models.cosmos import CosmosNanoAdapter  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

SIZE = 1024
palette = build(48)
adapter = CosmosNanoAdapter(f"{VOL}/models/cosmos3_nano", LORA, size=SIZE)

CLIPS = [
    ("dt0004_f1_30", "dancetrack0004", 1, 30),
    ("dt0004_gap8", "dancetrack0004", 300, 325),
    ("dt0005_gap16", "dancetrack0005", 548, 580),
]
out = {}
for name, seqname, start, end in CLIPS:
    try:
        seq = load_sequence(f"{VOL}/dancetrack/val/{seqname}")
        w, h = seq.image_size
        sx, sy = SIZE / w, SIZE / h
        frames = [_resize_frame(seq.frame_path(f), SIZE) for f in range(start, end + 1)]
        t0 = time.time()
        clean = chained_track(adapter, frames, palette=palette, K=48, min_area=300, alpha=0.5)
        pred = idmaps_to_boxes(clean, palette=palette, min_area=300)
        gt = []
        for i, f in enumerate(range(start, end + 1), start=1):
            for b in seq.boxes_at(f):
                gt.append(Box(i, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis))
        m = evaluate(gt, pred, iou_threshold=0.5)
        events = mine_occlusion(gt, strategy="absence", min_gap=5, max_gap=40)
        reacq = reacquisition_check(pred, gt, events, iou_thr=0.5)
        out[name] = {
            "frames": len(frames), "gt_tracks": len({b.track_id for b in gt}),
            "pred_tracks": len({b.track_id for b in pred}),
            "idf1": round(m["idf1"], 4), "mota": round(m["mota"], 4),
            "idsw": m["num_switches"], "det@0.5_proxy": round(1 - m["motp"], 3),
            "reacq_rate": reacq["reacq_rate"], "redetect_rate": reacq["redetect_rate"],
            "reacq_events": reacq["n"], "sec": round(time.time() - t0, 1),
        }
        log(f"{name}: {out[name]}")
    except Exception:
        out[name] = {"error": traceback.format_exc()[-600:]}
        log(f"{name} FAILED:\n{out[name]['error']}")
    with open(f"{RES}/p1_identity.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

log("P1-IDENTITY-EVAL-DONE -> " + RES)
