# Databricks notebook source
# P0 Stage-1 LoRA fine-tune: frame -> ID-map perception on Cosmos3-Nano (image mode,
# num_frames=1, frame conditions via understanding pathway, generate the ID-map).
# Flow-matching velocity loss. Logs smoothed loss + saves LoRA checkpoints + generates
# sample ID-maps from a held-out val frame (the real "is it learning" signal). All to volume.

import json
import os
import random
import shutil
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
    "peft", "safetensors", "sentencepiece", "pyyaml")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/cosmos3_nano"
DT = f"{VOL}/dancetrack"
# Mask source for the Stage-1 target ID-maps (§8.2). "sam_pseudo" = SAM 3.1 silhouettes
# (data/pseudo_masks.py, train splits only); "filled_box" = rectangles (the GATE-0 run).
MASK_SOURCE = "sam_pseudo"
PSEUDO_ROOT = f"{VOL}/pseudo_masks/dancetrack"
_sfx = "_sam" if MASK_SOURCE == "sam_pseudo" else ""
OUT = f"{VOL}/checkpoints/p0_stage1{_sfx}"  # own dir → keeps the filled-box run intact
os.makedirs(OUT, exist_ok=True)
# safetensors save_file mmaps the destination → fails on the blob FUSE mount (os err 11).
# Write checkpoints/samples to local disk, then copy to the volume with normal I/O.
LOCAL = f"/local_disk0/p0_stage1{_sfx}"
os.makedirs(LOCAL, exist_ok=True)


def copy_to_vol(local_dir, vol_dir):
    os.makedirs(vol_dir, exist_ok=True)
    for fn in os.listdir(local_dir):
        src = os.path.join(local_dir, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(vol_dir, fn))
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"

# pinned (mirror image_generators/cosmos/train.yaml)
MAX_STEPS = 6000        # backstop; early stopping usually ends sooner
EVAL_EVERY = 500        # validate on internal-val every N steps
PATIENCE = 3            # stop after this many evals with no val mean-IoU improvement
VAL_FPV = 3             # val frames sampled per internal-val video
SAMPLE_EVERY = 1000     # save a visual frame|GT|gen sample this often
LOG_EVERY = 25
GRAD_ACCUM = 4
LR = 1e-4
SIZE = 1024
# i2v spatial conditioning (validated by cond_sanity, RUN 865988846254198): num_frames=5
# -> 2 latent frames [clean RGB condition, denoised ID-map target] on the same 64x64 grid.
# The frame is the condition; the ID-map is generated at the last latent frame.
NUM_FRAMES = 5
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

from diffusers import Cosmos3OmniPipeline  # noqa: E402
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler  # noqa: E402
from peft import LoraConfig  # noqa: E402

from scipy.optimize import linear_sum_assignment  # noqa: E402

from gentrack.color_codec import decode  # noqa: E402
from gentrack.data.datasets import dancetrack_splits  # noqa: E402
from gentrack.data.pseudo_masks import (  # noqa: E402
    idmap_from_instance_map,
    load_instance_map,
    resize_instance_map,
)
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame, _scale_boxes  # noqa: E402
from gentrack.data.build_idmaps import render_idmap  # noqa: E402

res = {"losses": [], "smoothed": [], "samples": []}


def save_log():
    with open(os.path.join(OUT, "train_log.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)


pipe = Cosmos3OmniPipeline.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
tf = pipe.transformer
lin = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
targets = [n for n in ("to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj") if n in lin]
tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0))
if hasattr(tf, "enable_gradient_checkpointing"):
    tf.enable_gradient_checkpointing()
log(f"loaded + LoRA on {targets}")

sched = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
sched.set_timesteps(1000, device="cuda")
timesteps, sigmas = sched.timesteps, sched.sigmas

# data: flat index over the 35-video train split
data_cfg = None
import yaml  # noqa: E402
with open(os.environ["GENTRACK_WS"] + "/datasets/data.yaml") as f:
    data_cfg = yaml.safe_load(f)
dc = data_cfg["datasets"]["dancetrack"]
train_seqs, val_seqs, _ = dancetrack_splits(DT, dc["train_dirs"], dc["internal_val"], dc["test_dirs"])
K = int(data_cfg["palette"]["K"])  # 48
palette = build(K)
index = [(s, f) for s in train_seqs for f in range(1, s.num_frames + 1)]
random.shuffle(index)
log(f"train: {len(train_seqs)} videos, {len(index)} frames | val: {len(val_seqs)}")


def make_pair(seq, f):
    # Stage-1: per-frame colour assignment (distinct colour per person present),
    # not sequence-stable — avoids >K-1-id overflow on long videos. SAM silhouettes
    # come from the precomputed native-res instance map (resized NEAREST); filled_box
    # renders the GT boxes. Eval still uses GT boxes (hygiene), so this only swaps the
    # training target.
    w, h = seq.image_size
    frame = _resize_frame(seq.frame_path(f), SIZE)
    if MASK_SOURCE == "sam_pseudo":
        arr = load_instance_map(os.path.join(PSEUDO_ROOT, seq.name, f"{f:08d}.png"))
        idmap = idmap_from_instance_map(resize_instance_map(arr, SIZE), palette, K=K)
    else:
        boxes = _scale_boxes(seq.boxes_at(f), SIZE / w, SIZE / h)
        idmap = render_idmap(boxes, (SIZE, SIZE), palette, assignment=None, K=K)
    return frame, idmap


def capture_static(frame_pil):
    cap = {}

    class _Stop(Exception):
        pass

    def hook(m, a, k):
        cap["k"] = dict(k)
        raise _Stop()

    h = tf.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            pipe(prompt=PROMPT, image=frame_pil, num_frames=NUM_FRAMES, height=SIZE, width=SIZE,
                 num_inference_steps=1, guidance_scale=1.0,
                 generator=torch.Generator(device="cuda").manual_seed(0))
    except _Stop:
        pass
    finally:
        h.remove()
    return cap["k"]


def encode_idmap(idmap_np):
    x = torch.from_numpy(idmap_np).permute(2, 0, 1)[None, :, None].float()
    x = (x / 127.5 - 1.0).to("cuda", torch.bfloat16)
    with torch.no_grad():
        return pipe._encode_video(x)


def gen_sample(seq, f, step):
    frame, idmap_gt = make_pair(seq, f)
    tf.eval()
    try:
        with torch.no_grad():
            o = pipe(prompt=PROMPT, image=Image.fromarray(frame), num_frames=NUM_FRAMES, height=SIZE,
                     width=SIZE, num_inference_steps=20, guidance_scale=4.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
        gen = np.asarray(o.video[-1])  # frame 0 is the RGB condition; the ID-map is generated after
        ld = os.path.join(LOCAL, f"sample_step{step}")
        os.makedirs(ld, exist_ok=True)
        Image.fromarray(frame).save(f"{ld}/frame.png")
        Image.fromarray(idmap_gt).save(f"{ld}/gt.png")
        Image.fromarray(gen).save(f"{ld}/gen.png")
        vd = os.path.join(OUT, f"sample_step{step}")
        copy_to_vol(ld, vd)
        ninst = len(decode(gen, palette=palette, assignment=None, min_area=200))
        res["samples"].append({"step": step, "gen_decoded_instances": ninst, "dir": vd})
        log(f"  sample@{step}: gen decoded {ninst} instances -> {vd}")
    except Exception:
        res.setdefault("sample_errors", []).append(traceback.format_exc()[-500:])
    tf.train()


# validation set: VAL_FPV evenly-spaced frames from each internal-val video, with GT
# boxes scaled to the (square) training resolution to match decoded predictions.
val_eval = []
for s in val_seqs:
    w, h = s.image_size
    fs = np.linspace(1, s.num_frames, VAL_FPV + 2, dtype=int)[1:-1]
    for fnum in fs:
        frame = _resize_frame(s.frame_path(int(fnum)), SIZE)
        gtb = _scale_boxes(s.boxes_at(int(fnum)), SIZE / w, SIZE / h)
        val_eval.append((frame, gtb))
log(f"val eval frames: {len(val_eval)} (from {len(val_seqs)} videos)")


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def evaluate():
    """Mean box-IoU + detection@0.5 of decoded generations vs GT over the val frames."""
    tf.eval()
    ious, dets = [], []
    with torch.no_grad():
        for frame, gtb in val_eval:
            if not gtb:
                continue
            o = pipe(prompt=PROMPT, image=Image.fromarray(frame), num_frames=NUM_FRAMES,
                     height=SIZE, width=SIZE, num_inference_steps=20, guidance_scale=4.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
            pb = idmaps_to_boxes([np.asarray(o.video[-1])], palette=palette, min_area=300)
            if not pb:
                ious.append(0.0)
                dets.append(0.0)
                continue
            C = np.array([[box_iou(g.xywh, p.xywh) for p in pb] for g in gtb])
            r, c = linear_sum_assignment(-C)
            matched = [C[i, j] for i, j in zip(r, c)]
            ious.append(float(np.mean(matched)) if matched else 0.0)
            dets.append(sum(v >= 0.5 for v in matched) / len(gtb))
    tf.train()
    return {"mean_iou": round(float(np.mean(ious)), 4), "det@0.5": round(float(np.mean(dets)), 4)}


tf.train()
opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=LR)
res["val"] = []
best, best_step, patience_left = -1.0, 0, PATIENCE
window = []
t_start = time.time()
try:
    for step in range(MAX_STEPS):
        seq, f = index[step % len(index)]
        frame, idmap = make_pair(seq, f)
        k = capture_static(Image.fromarray(frame))
        rgb_lat = k["vision_tokens"][0][:, :, 0:1]          # clean RGB condition (latent frame 0)
        x0 = encode_idmap(idmap)                            # ID-map target latent (1,48,1,64,64)
        noise = torch.randn_like(x0)
        idx = int(torch.randint(0, len(timesteps), (1,)))
        t, sigma = timesteps[idx], sigmas[idx].to(x0.dtype)
        x_t = (1.0 - sigma) * x0 + sigma * noise
        vtarget = noise - x0
        n_noisy = int(k["vision_mse_loss_indexes"].numel())
        k["vision_tokens"] = [torch.cat([rgb_lat, x_t], dim=2).to(torch.bfloat16)]
        k["vision_timesteps"] = torch.full((n_noisy,), float(t), device="cuda", dtype=torch.bfloat16)
        out = tf(**k)
        pred = out[0]
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        pred_t = pred[:, :, -1:]                            # target = last latent frame (ID-map)
        loss = F.mse_loss(pred_t.float(), vtarget.reshape(pred_t.shape).float()) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            opt.step()
            opt.zero_grad()
        lv = float(loss) * GRAD_ACCUM
        res["losses"].append(round(lv, 4))
        window.append(lv)
        if len(window) > 100:
            window.pop(0)
        if step % LOG_EVERY == 0:
            sm = round(float(np.mean(window)), 4)
            res["smoothed"].append({"step": step, "loss": sm})
            log(f"step {step}/{MAX_STEPS} smoothed_loss {sm} ({(time.time()-t_start)/(step+1):.2f}s/step)")
            save_log()
        if (step + 1) % SAMPLE_EVERY == 0:
            gen_sample(val_seqs[0], 1, step + 1)
        if (step + 1) % EVAL_EVERY == 0:
            v = evaluate()
            v["step"] = step + 1
            res["val"].append(v)
            log(f"VAL@{step+1}: {v} | best {best:.4f}@{best_step}")
            if v["mean_iou"] > best:
                best, best_step, patience_left = v["mean_iou"], step + 1, PATIENCE
                lck = os.path.join(LOCAL, "lora_best")
                tf.save_pretrained(lck)
                copy_to_vol(lck, os.path.join(OUT, "lora_best"))
                res["best"] = {"step": best_step, "mean_iou": best, "det@0.5": v["det@0.5"]}
            else:
                patience_left -= 1
            save_log()
            if patience_left <= 0:
                log(f"early stop @ {step+1}: no val improvement for {PATIENCE} evals (best {best:.4f}@{best_step})")
                break
except Exception:
    res["error"] = traceback.format_exc()
    save_log()

res["total_min"] = round((time.time() - t_start) / 60, 1)
save_log()
log(f"P0-FINETUNE-DONE in {res['total_min']} min | best mean_iou {best:.4f}@{best_step} -> {OUT}")
