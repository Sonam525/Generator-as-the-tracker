# Databricks notebook source
# SAM 3.1 (facebook/sam3) smoke test — Mode 1, box-prompted, identity from GT.
# Verifies: (1) gated access with HF_TOKEN, (2) the transformers Sam3Model/Sam3Processor
# box-prompt API + post-process shape, (3) silhouette quality vs filled boxes on a few
# TRAIN frames. Saves a contact sheet (frame+GT boxes | frame+SAM masks | instance map)
# and stats to the volume. No full pass until this is reviewed.

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

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.pseudo_masks import (  # noqa: E402
    Sam3Masker,
    build_instance_map,
    mask_bbox,
    match_masks_to_boxes,
)
from gentrack.palette import build  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
RES = f"{VOL}/results/sam3_smoke"
os.makedirs(RES, exist_ok=True)
SEQ = f"{VOL}/dancetrack/train1/dancetrack0008"  # TRAIN split (hygiene ok)
FRAMES = [1, 40, 80]
palette = build(48)

out = {}
try:
    import torch

    masker = Sam3Masker(hf_token=HF_TOKEN)
    seq = load_sequence(SEQ)
    w, h = seq.image_size

    rows = []
    for fidx in FRAMES:
        img = np.asarray(Image.open(seq.frame_path(fidx)).convert("RGB"))
        boxes = seq.boxes_at(fidx)
        t0 = time.time()
        masks = masker.masks_for(img, boxes)
        dt = time.time() - t0

        # one-time raw-shape dump so the API return structure is on record
        if fidx == FRAMES[0]:
            inp = masker._proc(images=Image.fromarray(img),
                               input_boxes=[[[float(b.x), float(b.y), float(b.x + b.w),
                                              float(b.y + b.h)] for b in boxes]],
                               input_boxes_labels=[[1] * len(boxes)],
                               return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = masker._model(**inp)
            pp = masker._proc.post_process_instance_segmentation(
                o, threshold=masker.score_threshold, mask_threshold=masker.mask_threshold,
                target_sizes=inp.get("original_sizes").tolist())
            out["raw_result_type"] = str(type(pp[0]))
            out["raw_result_keys"] = list(pp[0].keys()) if isinstance(pp[0], dict) else "n/a"

        matched = match_masks_to_boxes(masks, boxes, iou_thr=0.5)
        ious, tightness = [], []
        for b in boxes:
            m = matched.get(b.track_id)
            if m is None:
                continue
            bb = mask_bbox(m)
            ax, ay, aw, ah = b.xywh
            ix0, iy0 = max(ax, bb[0]), max(ay, bb[1])
            ix1, iy1 = min(ax + aw, bb[0] + bb[2]), min(ay + ah, bb[1] + bb[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = aw * ah + bb[2] * bb[3] - inter
            ious.append(inter / union if union > 0 else 0.0)
            tightness.append(float(m.sum()) / (aw * ah) if aw * ah > 0 else 0.0)
        imap = build_instance_map(matched, boxes, (w, h))

        out[f"frame{fidx}"] = {
            "n_boxes": len(boxes), "n_masks": len(masks), "n_matched": len(matched),
            "match_rate": round(len(matched) / max(1, len(boxes)), 3),
            "mean_box_mask_iou": round(float(np.mean(ious)), 3) if ious else None,
            "mean_silhouette_tightness": round(float(np.mean(tightness)), 3) if tightness else None,
            "sec": round(dt, 2),
        }
        log(f"frame {fidx}: {out[f'frame{fidx}']}")

        # contact-sheet row: GT boxes | SAM masks | colorized instance map
        gt_img = Image.fromarray(img.copy())
        d = ImageDraw.Draw(gt_img)
        for b in boxes:
            d.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=(255, 0, 0), width=3)
        sam_img = img.copy()
        for tid, m in matched.items():
            sam_img[m.astype(bool)] = palette[(tid % 47) + 1]
        cmap = np.zeros((h, w, 3), dtype=np.uint8)
        for v in np.unique(imap):
            if v:
                cmap[imap == v] = palette[(int(v) % 47) + 1]
        row = np.concatenate([np.asarray(gt_img), sam_img, cmap], axis=1)
        rows.append(row)

    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).save(f"{RES}/contact_sheet.png")
    if "torch" in sys.modules:
        out["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out["status"] = "OK"
    log(f"SMOKE OK: {out.get('vram_gb')} GB, sheet -> {RES}/contact_sheet.png")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-1500:]
    log("SMOKE FAILED:\n" + out["error"])

with open(f"{RES}/smoke.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("SAM3-SMOKE-DONE -> " + RES)
