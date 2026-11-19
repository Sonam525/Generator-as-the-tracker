# Databricks notebook source
# MOT17 re-id eval (Option B, zero-shot perception). Same re-id pipeline validated on
# DanceTrack: conditioned-P0 (DanceTrack-trained) perceives each frame, IdentityBank
# associates regions by DINOv3 appearance + motion (frozen under occlusion), assigns stable
# canonical colours. Run on the SPARSE MOT17 val-halves (where zero-shot perception is
# usable), windowed, with low_visibility occlusion mining → IDF1/IDSW + RE-ACQUISITION over
# MANY events (the statistical power DanceTrack's 1-event clips lacked). The headline test:
# does appearance re-id push reacq above 0 on appearance-distinct pedestrians?

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

pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "-U", "transformers",
    "peft", "safetensors", "sentencepiece", "motmetrics")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gentrack.color_codec import decode, encode  # noqa: E402
from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.datasets import mot17_splits  # noqa: E402
from gentrack.data.mot_format import Box  # noqa: E402
from gentrack.eval.mot_metrics import detection_recall, evaluate, reacquisition_check  # noqa: E402
from gentrack.models.cosmos import CosmosNanoAdapter  # noqa: E402
from gentrack.models.identity_bank import IdentityBank  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
LORA = f"{VOL}/checkpoints/p0_stage1_sam/lora_best"
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RES = f"{VOL}/results/mot17_reid_eval_v2"   # de-confound run (motion-primary matching)
os.makedirs(RES, exist_ok=True)
SIZE, MIN_AREA, MAX_FRAMES = 1024, 200, 100
# v2 de-confound (PRINCIPLED, not eval-tuned): v1 hit the 47-colour cap = matching failed →
# over-fragmentation. Motion IoU is reliable + scale-free frame-to-frame; DINOv3 appearance
# is weak (diagnostic top1 0.35). So weight motion primary; permissive threshold still lets
# a strong appearance match (cos>=0.5) re-bind across a gap (iou=0): cost=0.3(1-cos)+0.7 <= 0.85.
W_APP, W_MOTION, MATCH_THR = 0.3, 0.7, 0.85
SEQS = ["MOT17-05", "MOT17-10"]   # best-detection + most re-acq events (fair, cheap read)
palette = build(48)
adapter = CosmosNanoAdapter(f"{VOL}/models/cosmos3_nano", LORA, size=SIZE)

from transformers import AutoImageProcessor, AutoModel  # noqa: E402
_dproc = AutoImageProcessor.from_pretrained(DINO_ID, token=HF_TOKEN)
_dino = AutoModel.from_pretrained(DINO_ID, token=HF_TOKEN).to("cuda").eval()


def appearance(frame, box):
    x, y, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    crop = frame[y:y + bh, x:x + bw]
    if crop.size == 0:
        return np.zeros(768, np.float32)
    inp = _dproc(images=Image.fromarray(crop), return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = _dino(**inp)
    return o.last_hidden_state[0, 0].float().cpu().numpy()


def reid_track(frames):
    bank = IdentityBank(momentum=0.9, max_misses=50)
    clean = []
    for fi, frame in enumerate(frames, start=1):
        masks = list(decode(adapter.generate(frame), palette=palette, assignment=None, min_area=MIN_AREA).values())
        if not masks:
            clean.append(np.zeros((SIZE, SIZE, 3), np.uint8))
            continue
        dets = []
        for m in masks:
            ys, xs = np.where(m)
            box = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
            dets.append((appearance(frame, box), box))
        colors = bank.assign(dets, fi, w_app=W_APP, w_motion=W_MOTION, match_thr=MATCH_THR, K=48)
        keep = [(m, c) for m, c in zip(masks, colors) if c is not None]
        if keep:
            mm, cc = zip(*keep)
            clean.append(encode(list(mm), list(cc), palette=palette, assignment={c: c for c in cc}))
        else:
            clean.append(np.zeros((SIZE, SIZE, 3), np.uint8))
    return clean


_, val = mot17_splits(f"{VOL}/MOT17", "train")
val_by = {s.name.split("_")[0]: s for s in val}

out = {"params": {"w_app": W_APP, "w_motion": W_MOTION, "match_thr": MATCH_THR, "max_frames": MAX_FRAMES}}
for sname in SEQS:
    try:
        seq = val_by.get(sname)
        if seq is None:
            out[sname] = {"error": "not found"}
            continue
        w, h = seq.image_size
        sx, sy = SIZE / w, SIZE / h
        nf = min(MAX_FRAMES, seq.num_frames)
        frames = [_resize_frame(seq.frame_path(f), SIZE) for f in range(1, nf + 1)]
        t0 = time.time()
        clean = reid_track(frames)
        pred = idmaps_to_boxes(clean, palette=palette, min_area=MIN_AREA)
        gt = []
        for f in range(1, nf + 1):
            for b in seq.boxes_at(f):
                gt.append(Box(f, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis))
        m5 = evaluate(gt, pred, iou_threshold=0.5)
        m3 = evaluate(gt, pred, iou_threshold=0.3)
        rec = detection_recall(gt, pred, iou_thrs=(0.5, 0.3, 0.1))
        events = mine_occlusion(gt, strategy="low_visibility", vis_threshold=0.5, min_gap=5, max_gap=40)
        reacq5 = reacquisition_check(pred, gt, events, iou_thr=0.5)
        reacq3 = reacquisition_check(pred, gt, events, iou_thr=0.3)
        out[sname] = {
            "frames": nf, "gt_tracks": len({b.track_id for b in gt}),
            "pred_tracks": len({b.track_id for b in pred}),
            "idf1@0.5": round(m5["idf1"], 4), "idf1@0.3": round(m3["idf1"], 4),
            "idsw@0.5": m5["num_switches"], "idsw@0.3": m3["num_switches"],
            "recall@0.5": rec["recall@0.5"], "recall@0.3": rec["recall@0.3"], "recall@0.1": rec["recall@0.1"],
            "reacq_events": reacq5["n"],
            "reacq@0.5": reacq5["reacq_rate"], "redetect@0.5": reacq5["redetect_rate"],
            "reacq@0.3": reacq3["reacq_rate"], "redetect@0.3": reacq3["redetect_rate"],
            "sec": round(time.time() - t0, 1),
        }
        log(f"{sname}: {out[sname]}")
    except Exception:
        out[sname] = {"error": traceback.format_exc()[-600:]}
        log(f"{sname} FAILED: {out[sname]['error']}")
    with open(f"{RES}/eval.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

log("MOT17-REID-EVAL-DONE -> " + RES)
