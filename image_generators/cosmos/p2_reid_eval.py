# Databricks notebook source
# P2 Option-3 (re-id pivot) eval — NO new training. The conditioned-P0 generator perceives
# each frame (fresh, drift-free); regions are decoded; each gets a VAE-latent appearance
# embedding + box; the IdentityBank associates them across frames by APPEARANCE + MOTION
# (frozen under occlusion → re-acquisition), assigning stable canonical colours rendered
# back into the ID-map. Head-to-head with the no-state floor (results/p2_baseline_cond) on
# the same 3 occlusion clips, same metrics @IoU 0.3 & 0.5. If this lifts IDF1 / pushes
# reacq>0 over the heuristic floor, the pivot works with zero training. GPU only, no API.

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
from PIL import Image  # noqa: E402

from gentrack.color_codec import decode, encode  # noqa: E402
from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.mot_format import Box  # noqa: E402
from gentrack.eval.mot_metrics import detection_recall, evaluate, reacquisition_check  # noqa: E402
from gentrack.models.cosmos import CosmosNanoAdapter  # noqa: E402
from gentrack.models.identity_bank import IdentityBank  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
LORA = f"{VOL}/checkpoints/p0_stage1_sam/lora_best"   # conditioned (i2v) + SAM-silhouette perceiver
APP_SOURCE = "dino"   # "dino" (DINOv3 crop, v2) | "vae" (pooled latent, v1)
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RES = f"{VOL}/results/p2_reid_eval_{APP_SOURCE}"
os.makedirs(RES, exist_ok=True)
SIZE, MIN_AREA = 1024, 300
# UNPINNED re-id params (propose; pin in image_generators/cosmos/train.yaml). DanceTrack appearance is
# ambiguous so motion carries frame-to-frame; appearance must still re-bind across gaps.
W_APP, W_MOTION, MATCH_THR = 0.5, 0.5, 0.6
palette = build(48)
adapter = CosmosNanoAdapter(f"{VOL}/models/cosmos3_nano", LORA, size=SIZE)

_dino = _dproc = None
if APP_SOURCE == "dino":
    import torch  # noqa: E402
    from transformers import AutoImageProcessor, AutoModel  # noqa: E402
    _dproc = AutoImageProcessor.from_pretrained(DINO_ID, token=HF_TOKEN)
    _dino = AutoModel.from_pretrained(DINO_ID, token=HF_TOKEN).to("cuda").eval()


def appearance(frame, mask, box, lat):
    """Per-region appearance embedding: DINOv3 CLS over the crop (v2) or pooled VAE latent
    over the region (v1). The bank is dim-agnostic (cosine), so the dim can differ."""
    if APP_SOURCE == "dino":
        x, y, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop = frame[y:y + bh, x:x + bw]
        if crop.size == 0:
            return np.zeros(768, np.float32)
        inp = _dproc(images=Image.fromarray(crop), return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = _dino(**inp)
        return o.last_hidden_state[0, 0].float().cpu().numpy()
    zh, zw = lat.shape[1], lat.shape[2]
    m64 = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((zw, zh), Image.NEAREST)).astype(bool)
    return lat[:, m64].mean(axis=1) if m64.any() else np.zeros(lat.shape[0], np.float32)


def regions(raw):
    """Decode the generator's (arbitrary-colour) map into per-region masks = detections."""
    inst = decode(raw, palette=palette, assignment=None, min_area=MIN_AREA)
    return list(inst.values())


def reid_track(frames):
    bank = IdentityBank(momentum=0.9, max_misses=50)
    clean_maps = []
    for fi, frame in enumerate(frames, start=1):
        masks = regions(adapter.generate(frame))
        if not masks:
            clean_maps.append(np.zeros((SIZE, SIZE, 3), np.uint8))
            continue
        lat = adapter.frame_latent(frame) if APP_SOURCE == "vae" else None
        dets = []
        for m in masks:
            ys, xs = np.where(m)
            box = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
                   float(ys.max() - ys.min() + 1))
            dets.append((appearance(frame, m, box, lat), box))
        colors = bank.assign(dets, fi, w_app=W_APP, w_motion=W_MOTION, match_thr=MATCH_THR, K=48)
        keep = [(m, c) for m, c in zip(masks, colors) if c is not None]
        if keep:
            mm, cc = zip(*keep)
            clean_maps.append(encode(list(mm), list(cc), palette=palette, assignment={c: c for c in cc}))
        else:
            clean_maps.append(np.zeros((SIZE, SIZE, 3), np.uint8))
    return clean_maps


CLIPS = [
    ("dt0004_f1_30", "dancetrack0004", 1, 30),
    ("dt0004_gap8", "dancetrack0004", 300, 325),
    ("dt0005_gap16", "dancetrack0005", 548, 580),
]
out = {"params": {"w_app": W_APP, "w_motion": W_MOTION, "match_thr": MATCH_THR}}
for name, seqname, start, end in CLIPS:
    try:
        seq = load_sequence(f"{VOL}/dancetrack/val/{seqname}")
        w, h = seq.image_size
        sx, sy = SIZE / w, SIZE / h
        frames = [_resize_frame(seq.frame_path(f), SIZE) for f in range(start, end + 1)]
        t0 = time.time()
        clean = reid_track(frames)
        pred = idmaps_to_boxes(clean, palette=palette, min_area=MIN_AREA)
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
            "recall@0.5": rec["recall@0.5"], "recall@0.3": rec["recall@0.3"], "recall@0.1": rec["recall@0.1"],
            "reacq@0.5": reacq5["reacq_rate"], "redetect@0.5": reacq5["redetect_rate"],
            "reacq@0.3": reacq3["reacq_rate"], "redetect@0.3": reacq3["redetect_rate"],
            "reacq_events": reacq5["n"], "sec": round(time.time() - t0, 1),
        }
        log(f"{name}: {out[name]}")
    except Exception:
        out[name] = {"error": traceback.format_exc()[-600:]}
        log(f"{name} FAILED:\n{out[name]['error']}")
    with open(f"{RES}/eval.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

log("P2-REID-EVAL-DONE -> " + RES)
