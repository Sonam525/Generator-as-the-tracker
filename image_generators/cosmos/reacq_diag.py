# Databricks notebook source
# Re-acquisition diagnostic: is APPEARANCE the bottleneck, and would a strong encoder fix it?
# For each mined occlusion event (GT track G: visible -> gap -> reappears), embed G at the
# reappearance frame and every candidate track at the pre-gap frame, and ask: is G's OWN
# prior track the distinctively-highest cosine match among candidates? Tested with BOTH the
# current VAE-latent appearance AND DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m).
# GT boxes only (decoupled from perception); INTERNAL-VAL videos (hygienic, more events).
# top1-rate ~ chance + ~0 margin => appearance ambiguous (fundamental); high top1 + margin
# for DINOv3 => appearance was the bottleneck and v2 (DINOv3) will lift re-acq.

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
    "safetensors", "sentencepiece")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/cosmos3_nano"
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
RES = f"{VOL}/results/reacq_diag"
os.makedirs(RES, exist_ok=True)
SIZE = 1024
INTERNAL_VAL = ["dancetrack0008", "dancetrack0029", "dancetrack0051", "dancetrack0066", "dancetrack0096"]
MAX_EVENTS = 50


def seq_dir(name):
    for d in ("train1", "train2"):
        p = f"{VOL}/dancetrack/{d}/{name}"
        if os.path.isdir(p):
            return p
    return None


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _crop(img_np, box):
    h, w = img_np.shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
    if x1 <= x0 or y1 <= y0:
        return None
    return img_np[y0:y1, x0:x1]


out = {}
try:
    from diffusers import Cosmos3OmniPipeline
    from transformers import AutoImageProcessor, AutoModel

    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
    dproc = AutoImageProcessor.from_pretrained(DINO_ID, token=HF_TOKEN)
    dino = AutoModel.from_pretrained(DINO_ID, token=HF_TOKEN).to("cuda").eval()
    log("loaded Cosmos VAE + DINOv3")

    _vae_cache = {}

    def vae_latent(seq, fnum):
        key = (seq.name, fnum)
        if key not in _vae_cache:
            im = np.asarray(Image.open(seq.frame_path(fnum)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
            x = torch.from_numpy(im).permute(2, 0, 1)[None, :, None].float()
            x = (x / 127.5 - 1.0).to("cuda", torch.bfloat16)
            with torch.no_grad():
                _vae_cache[key] = pipe._encode_video(x)[0, :, 0].float().cpu().numpy()  # (z,64,64)
        return _vae_cache[key]

    def emb_vae(seq, fnum, box):  # box in native px
        w, h = seq.image_size
        lat = vae_latent(seq, fnum)
        z, gh, gw = lat.shape
        x, y, bw, bh = box
        x0, y0 = int(x / w * gw), int(y / h * gh)
        x1, y1 = max(x0 + 1, int((x + bw) / w * gw)), max(y0 + 1, int((y + bh) / h * gh))
        return _unit(lat[:, y0:y1, x0:x1].reshape(z, -1).mean(axis=1))

    def emb_dino(seq, fnum, box):
        img = np.asarray(Image.open(seq.frame_path(fnum)).convert("RGB"))
        crop = _crop(img, box)
        if crop is None or crop.size == 0:
            return None
        inp = dproc(images=Image.fromarray(crop), return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = dino(**inp)
        cls = o.last_hidden_state[0, 0].float().cpu().numpy()
        return _unit(cls)

    def rank_and_margin(q, cands, true_tid):
        # cands: {tid: emb}; returns (rank of true by cosine desc, margin true - best-distractor)
        cos = {tid: float(np.dot(q, e)) for tid, e in cands.items() if e is not None}
        if true_tid not in cos:
            return None
        order = sorted(cos, key=lambda t: cos[t], reverse=True)
        rank = order.index(true_tid) + 1
        others = [cos[t] for t in cos if t != true_tid]
        margin = cos[true_tid] - (max(others) if others else -1.0)
        return rank, margin, len(cos)

    events_used = 0
    rec = {"vae": {"top1": 0, "margins": [], "n": 0}, "dino": {"top1": 0, "margins": [], "n": 0}}
    gaps = []
    for name in INTERNAL_VAL:
        d = seq_dir(name)
        if d is None:
            continue
        seq = load_sequence(d)
        evs = mine_occlusion(seq.gt, strategy="absence", min_gap=5, max_gap=40)
        for e in evs:
            if events_used >= MAX_EVENTS:
                break
            before = {b.track_id: b for b in seq.boxes_at(e.before_frame)}
            after_g = [b for b in seq.boxes_at(e.after_frame) if b.track_id == e.track_id]
            if e.track_id not in before or not after_g or len(before) < 2:
                continue
            gb = after_g[0]
            for enc, embf in (("vae", emb_vae), ("dino", emb_dino)):
                q = embf(seq, e.after_frame, gb.xywh)
                cands = {tid: embf(seq, e.before_frame, b.xywh) for tid, b in before.items()}
                if q is None:
                    continue
                r = rank_and_margin(q, cands, e.track_id)
                if r is None:
                    continue
                rank, margin, ncand = r
                rec[enc]["n"] += 1
                rec[enc]["top1"] += int(rank == 1)
                rec[enc]["margins"].append(margin)
            gaps.append(e.gap)
            events_used += 1
        if events_used >= MAX_EVENTS:
            break

    for enc in ("vae", "dino"):
        r = rec[enc]
        out[enc] = {
            "n_events": r["n"],
            "top1_rate": round(r["top1"] / r["n"], 3) if r["n"] else None,
            "mean_margin": round(float(np.mean(r["margins"])), 4) if r["margins"] else None,
            "frac_positive_margin": round(float(np.mean([m > 0 for m in r["margins"]])), 3) if r["margins"] else None,
        }
    out["n_events"] = events_used
    out["mean_gap"] = round(float(np.mean(gaps)), 1) if gaps else None
    out["mean_candidates"] = "see per-event; >=2 tracks/event required"
    out["status"] = "OK"
    log(f"RESULTS: {out}")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-2500:]
    log("REACQ-DIAG FAILED:\n" + out["error"])

with open(f"{RES}/diag.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("REACQ-DIAG-DONE -> " + RES)
