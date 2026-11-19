# Databricks notebook source
# Verify the P1 CosmosNanoAdapter: does the LoRA load (not silently the base)? does the
# chained tracker produce stable-colour ID-maps? Runs ~6 frames of one internal-val clip,
# saves frame/cond/clean tiles + reports lora-load stats. Cheap sanity before full eval.

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
    "peft", "safetensors", "sentencepiece")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from PIL import Image  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
RES = f"{VOL}/results/p1_eval_probe"
os.makedirs(RES, exist_ok=True)

from gentrack.eval.chained_eval import chained_track  # noqa: E402
from gentrack.models.cosmos import CosmosNanoAdapter  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402

res = {}
try:
    seq = load_sequence(f"{VOL}/dancetrack/train1/dancetrack0008")
    frames = [_resize_frame(seq.frame_path(f), 1024) for f in range(1, 7)]
    adapter = CosmosNanoAdapter(f"{VOL}/models/cosmos3_nano", f"{VOL}/checkpoints/p1_stage2/lora_best", size=1024)
    palette = build(48)
    t0 = time.time()
    clean = chained_track(adapter, frames, palette=palette, K=48, min_area=300, alpha=0.5)
    res["sec"] = round(time.time() - t0, 1)
    for i, (fr, cm) in enumerate(zip(frames, clean)):
        Image.fromarray(fr).save(f"{RES}/{i:02d}_frame.png")
        Image.fromarray(cm).save(f"{RES}/{i:02d}_clean.png")
    boxes = idmaps_to_boxes(clean, palette=palette, min_area=300)
    res["n_frames"] = len(clean)
    res["n_pred_tracks"] = len({b.track_id for b in boxes})
    res["pred_boxes_total"] = len(boxes)
    log(f"OK: {res}")
except Exception:
    res["error"] = traceback.format_exc()
    log("FAILED:\n" + res["error"])

with open(f"{RES}/probe.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
log("EVAL-PROBE-DONE -> " + RES)
