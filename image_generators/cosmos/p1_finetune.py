# Databricks notebook source
# P2 step 4a — Stage-2 TEMPORAL trainer (Option C-lite), CONDITIONED. Fixes the void
# unconditioned P1: uses the validated i2v packing (condition latent = clean, target =
# denoised) with condition = composite(new_frame, prev cleaned ID-map) and SAM-silhouette
# SEQUENCE-STABLE targets (same dancer = same colour all video). The only "memory" is the
# previous ID-map's pixels in the condition (no learned bank) — the temporal arm that
# should cut the ~30 intra-clip ID-switches of the no-state floor. Flow-matching velocity
# loss; teacher-forced val box-IoU for early stopping. Chained IDF1/IDSW/reacq (the real
# verdict) is measured separately by the chained eval. Writes to checkpoints/p1_stage2_cond.

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
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/cosmos3_nano"
OUT = f"{VOL}/checkpoints/p1_stage2_cond"
DT = f"{VOL}/dancetrack"
PSEUDO = f"{VOL}/pseudo_masks/dancetrack"
os.makedirs(OUT, exist_ok=True)
LOCAL = "/local_disk0/p1_stage2_cond"
os.makedirs(LOCAL, exist_ok=True)
# Must match the adapter / P0 prompt (CosmosNanoAdapter.DEFAULT_PROMPT).
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"

MAX_STEPS, EVAL_EVERY, PATIENCE, VAL_FPV = 6000, 500, 3, 3
SAMPLE_EVERY, LOG_EVERY, GRAD_ACCUM, LR, SIZE, SEED = 1000, 25, 4, 1e-4, 1024, 0
NUM_FRAMES = 5       # i2v: condition latent (frame 0) + ID-map target (frame 1)
STRIDES = (1, 2, 4)
ALPHA = 0.5          # composite blend of prev ID-map onto the new frame
random.seed(SEED)
torch.manual_seed(SEED)


def copy_to_vol(local_dir, vol_dir):
    os.makedirs(vol_dir, exist_ok=True)
    for fn in os.listdir(local_dir):
        src = os.path.join(local_dir, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(vol_dir, fn))


from diffusers import Cosmos3OmniPipeline  # noqa: E402
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler  # noqa: E402
from peft import LoraConfig  # noqa: E402

from gentrack.data.datasets import dancetrack_splits  # noqa: E402
from gentrack.data.build_idmaps import sequence_assignment  # noqa: E402
from gentrack.data.pseudo_masks import (  # noqa: E402
    idmap_from_instance_map,
    load_instance_map,
    resize_instance_map,
)
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.pipeline.track import make_composite  # noqa: E402
from gentrack.train.stage1_data import _resize_frame, _scale_boxes  # noqa: E402

res = {"losses": [], "smoothed": [], "val": [], "samples": []}


def save_log():
    with open(os.path.join(OUT, "train_log.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


pipe = Cosmos3OmniPipeline.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
tf = pipe.transformer
lin = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
targets = [n for n in ("to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj") if n in lin]
tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0))
if hasattr(tf, "enable_gradient_checkpointing"):
    tf.enable_gradient_checkpointing()
log("loaded + LoRA")

sched = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
sched.set_timesteps(1000, device="cuda")
timesteps, sigmas = sched.timesteps, sched.sigmas

with open(os.environ["GENTRACK_WS"] + "/datasets/data.yaml") as f:
    dc = yaml.safe_load(f)["datasets"]["dancetrack"]
train_seqs, val_seqs, _ = dancetrack_splits(DT, dc["train_dirs"], dc["internal_val"], dc["test_dirs"])
K = 48
palette = build(K)
assign_cache = {s.name: sequence_assignment(s.track_ids, K) for s in train_seqs}
index = [(s, f, st) for s in train_seqs for st in STRIDES for f in range(1, s.num_frames - st + 1)]
random.shuffle(index)
log(f"train pairs: {len(index)} from {len(train_seqs)} videos | val {len(val_seqs)}")


def sam_idmap(seq, fr, a):
    arr = load_instance_map(f"{PSEUDO}/{seq.name}/{fr:08d}.png")
    return idmap_from_instance_map(resize_instance_map(arr, SIZE), palette, a)


def make_temporal(seq, f, stride):
    """(composite(new_frame, prev SAM idmap), target SAM idmap), sequence-stable colours."""
    a = assign_cache[seq.name]
    prev_idmap = sam_idmap(seq, f, a)
    new_frame = _resize_frame(seq.frame_path(f + stride), SIZE)
    target = sam_idmap(seq, f + stride, a)
    return make_composite(new_frame, prev_idmap, alpha=ALPHA), target


# teacher-forced val: composite(frame_{t+1}, SAM idmap_t) vs GT boxes_{t+1} (hygiene: GT boxes)
val_eval = []
for s in val_seqs:
    w, h = s.image_size
    sx, sy = SIZE / w, SIZE / h
    a = sequence_assignment(s.track_ids, K)
    fs = np.linspace(1, s.num_frames - 1, VAL_FPV + 2, dtype=int)[1:-1]
    for fnum in fs:
        prev = sam_idmap(s, int(fnum), a)
        nf = _resize_frame(s.frame_path(int(fnum) + 1), SIZE)
        val_eval.append((make_composite(nf, prev, alpha=ALPHA),
                         _scale_boxes(s.boxes_at(int(fnum) + 1), sx, sy)))
log(f"val eval pairs: {len(val_eval)}")


def capture_static(cond_pil):
    cap = {}

    class _Stop(Exception):
        pass

    def hook(m, a, k):
        cap["k"] = dict(k)
        raise _Stop()

    h = tf.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            pipe(prompt=PROMPT, image=cond_pil, num_frames=NUM_FRAMES, height=SIZE, width=SIZE,
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


def evaluate():
    tf.eval()
    ious, dets = [], []
    with torch.no_grad():
        for cond, gtb in val_eval:
            if not gtb:
                continue
            o = pipe(prompt=PROMPT, image=Image.fromarray(cond), num_frames=NUM_FRAMES, height=SIZE,
                     width=SIZE, num_inference_steps=20, guidance_scale=4.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
            pb = idmaps_to_boxes([np.asarray(o.video[-1])], palette=palette, min_area=300)
            if not pb:
                ious.append(0.0)
                dets.append(0.0)
                continue
            C = np.array([[box_iou(g.xywh, p.xywh) for p in pb] for g in gtb])
            r, c = linear_sum_assignment(-C)
            m = [C[i, j] for i, j in zip(r, c)]
            ious.append(float(np.mean(m)) if m else 0.0)
            dets.append(sum(v >= 0.5 for v in m) / len(gtb))
    tf.train()
    return {"mean_iou": round(float(np.mean(ious)), 4), "det@0.5": round(float(np.mean(dets)), 4)}


def gen_sample(step):
    cond, target = make_temporal(*index[0])
    tf.eval()
    try:
        with torch.no_grad():
            o = pipe(prompt=PROMPT, image=Image.fromarray(cond), num_frames=NUM_FRAMES, height=SIZE,
                     width=SIZE, num_inference_steps=20, guidance_scale=4.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
        ld = os.path.join(LOCAL, f"sample_step{step}")
        os.makedirs(ld, exist_ok=True)
        Image.fromarray(cond).save(f"{ld}/cond.png")
        Image.fromarray(target).save(f"{ld}/target.png")
        Image.fromarray(np.asarray(o.video[-1])).save(f"{ld}/gen.png")
        copy_to_vol(ld, os.path.join(OUT, f"sample_step{step}"))
    except Exception:
        res.setdefault("sample_errors", []).append(traceback.format_exc()[-400:])
    tf.train()


tf.train()
opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=LR)
best, best_step, patience_left = -1.0, 0, PATIENCE
window = []
t_start = time.time()
try:
    for step in range(MAX_STEPS):
        cond, target = make_temporal(*index[step % len(index)])
        k = capture_static(Image.fromarray(cond))
        cond_lat = k["vision_tokens"][0][:, :, 0:1]      # clean composite condition (frame 0)
        x0 = encode_idmap(target)                        # target ID-map latent (frame 1)
        noise = torch.randn_like(x0)
        idx = int(torch.randint(0, len(timesteps), (1,)))
        t, sigma = timesteps[idx], sigmas[idx].to(x0.dtype)
        x_t = (1.0 - sigma) * x0 + sigma * noise
        vtarget = noise - x0
        n_noisy = int(k["vision_mse_loss_indexes"].numel())
        k["vision_tokens"] = [torch.cat([cond_lat, x_t], dim=2).to(torch.bfloat16)]
        k["vision_timesteps"] = torch.full((n_noisy,), float(t), device="cuda", dtype=torch.bfloat16)
        out = tf(**k)
        pred = out[0]
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        pred_t = pred[:, :, -1:]                          # target = last latent frame
        loss = F.mse_loss(pred_t.float(), vtarget.reshape(pred_t.shape).float()) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            opt.step()
            opt.zero_grad()
        res["losses"].append(round(float(loss) * GRAD_ACCUM, 4))
        window.append(float(loss) * GRAD_ACCUM)
        if len(window) > 100:
            window.pop(0)
        if step % LOG_EVERY == 0:
            res["smoothed"].append({"step": step, "loss": round(float(np.mean(window)), 4)})
            log(f"step {step}/{MAX_STEPS} loss {res['smoothed'][-1]['loss']} ({(time.time()-t_start)/(step+1):.2f}s/step)")
            save_log()
        if (step + 1) % SAMPLE_EVERY == 0:
            gen_sample(step + 1)
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
                log(f"early stop @ {step+1} (best {best:.4f}@{best_step})")
                break
except Exception:
    res["error"] = traceback.format_exc()
    save_log()

res["total_min"] = round((time.time() - t_start) / 60, 1)
save_log()
log(f"P2-TEMPORAL-DONE {res['total_min']}min | best mean_iou {best:.4f}@{best_step}")
