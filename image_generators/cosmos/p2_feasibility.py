# Databricks notebook source
# P2 feasibility probe: the FAIR no-state floor — RGB-only perception + heuristic
# (IoU/colour) association via CanonicalTracker, which CANNOT copy-collapse (every frame
# re-perceives; previous map is never fed back). Uses the P0 per-frame perceiver LoRA on
# the 3 P-A-matched clips. Reports IDF1/IDSW/MOTA + re-acquisition + identity-FREE
# detection recall@{0.5,0.3,0.1}. This is the go/no-go input for building the P2 embedding
# bank: if detection recall is ~0 even at IoU 0.3, no memory mechanism can lift IDF1.
# GPU only, no API.

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
LORA = f"{VOL}/checkpoints/p0_stage1_sam/lora_best"   # conditioned (i2v) + SAM-silhouette P0
RES = f"{VOL}/results/p2_baseline_cond"               # real no-state floor (post conditioning fix)
os.makedirs(RES, exist_ok=True)

from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.mot_format import Box  # noqa: E402
from gentrack.eval.chained_eval import perframe_track  # noqa: E402
from gentrack.eval.mot_metrics import detection_recall, evaluate, reacquisition_check  # noqa: E402
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
        clean = perframe_track(adapter, frames, palette=palette, K=48, min_area=300)
        pred = idmaps_to_boxes(clean, palette=palette, min_area=300)
        gt = []
        for i, f in enumerate(range(start, end + 1), start=1):
            for b in seq.boxes_at(f):
                gt.append(Box(i, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis))
        m5 = evaluate(gt, pred, iou_threshold=0.5)
        m3 = evaluate(gt, pred, iou_threshold=0.3)
        rec = detection_recall(gt, pred, iou_thrs=(0.5, 0.3, 0.1))
        events = mine_occlusion(gt, strategy="absence", min_gap=5, max_gap=40)
        reacq5 = reacquisition_check(pred, gt, events, iou_thr=0.5)
        reacq3 = reacquisition_check(pred, gt, events, iou_thr=0.3)
        out[name] = {
            "frames": len(frames), "gt_tracks": len({b.track_id for b in gt}),
            "pred_tracks": len({b.track_id for b in pred}),
            "idf1@0.5": round(m5["idf1"], 4), "idf1@0.3": round(m3["idf1"], 4),
            "mota@0.5": round(m5["mota"], 4), "mota@0.3": round(m3["mota"], 4),
            "idsw@0.5": m5["num_switches"], "idsw@0.3": m3["num_switches"],
            "recall@0.5": rec["recall@0.5"], "recall@0.3": rec["recall@0.3"],
            "recall@0.1": rec["recall@0.1"],
            "reacq@0.5": reacq5["reacq_rate"], "redetect@0.5": reacq5["redetect_rate"],
            "reacq@0.3": reacq3["reacq_rate"], "redetect@0.3": reacq3["redetect_rate"],
            "reacq_events": reacq5["n"], "sec": round(time.time() - t0, 1),
        }
        log(f"{name}: {out[name]}")
    except Exception:
        out[name] = {"error": traceback.format_exc()[-600:]}
        log(f"{name} FAILED:\n{out[name]['error']}")
    with open(f"{RES}/p2_feas.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

log("P2-FEASIBILITY-DONE -> " + RES)
