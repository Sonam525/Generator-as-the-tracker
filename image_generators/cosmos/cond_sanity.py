# Databricks notebook source
# Conditioning sanity (Option A, i2v spatial packing): prove the frame is actually used,
# before any full P0 run. Pack RGB as the CLEAN condition latent (frame 0) and the ID-map
# as the DENOISED target (frame 1) on the same 64x64 grid, via num_frames=5 (-> 2 latent
# frames). Overfit TWO (frame, idmap) pairs that share the IDENTICAL prompt; then generate
# each. If conditioning works, each frame reproduces ITS OWN idmap (the model must read the
# frame to tell them apart, since the prompt is identical). If it's still text-only, it
# cannot. Decisive + cheap (~15 min).

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
    "peft", "safetensors", "sentencepiece", "scipy")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.pseudo_masks import (  # noqa: E402
    idmap_from_instance_map,
    load_instance_map,
    resize_instance_map,
)
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/cosmos3_nano"
PSEUDO = f"{VOL}/pseudo_masks/dancetrack"
RES = f"{VOL}/results/cond_sanity"
os.makedirs(RES, exist_ok=True)
SIZE = 1024
NUM_FRAMES = 5            # -> 2 latent frames: [cond RGB, target ID-map]
STEPS = 300
LR = 1e-4
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"
PAIRS = [("dancetrack0008", 1), ("dancetrack0008", 400)]  # same prompt, different layouts
palette = build(48)

import torch  # noqa: E402,F811
from diffusers import Cosmos3OmniPipeline  # noqa: E402
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler  # noqa: E402
from peft import LoraConfig  # noqa: E402

out = {"pairs": [p[0] + ":" + str(p[1]) for p in PAIRS]}


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mean_iou(pred_boxes, gt_boxes):
    if not gt_boxes:
        return None
    if not pred_boxes:
        return 0.0
    C = np.array([[box_iou(g.xywh, p.xywh) for p in pred_boxes] for g in gt_boxes])
    r, c = linear_sum_assignment(-C)
    return float(np.mean([C[i, j] for i, j in zip(r, c)]))


try:
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
    tf = pipe.transformer
    lin = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
    targets = [n for n in ("to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj") if n in lin]
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0))
    if hasattr(tf, "enable_gradient_checkpointing"):
        tf.enable_gradient_checkpointing()
    sched = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    sched.set_timesteps(1000, device="cuda")
    timesteps, sigmas = sched.timesteps, sched.sigmas

    def capture_i2v(frame_np):
        cap = {}

        class _Stop(Exception):
            pass

        def hook(m, a, k):
            cap["k"] = dict(k)
            raise _Stop()

        h = tf.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                pipe(prompt=PROMPT, image=Image.fromarray(frame_np), num_frames=NUM_FRAMES,
                     height=SIZE, width=SIZE, num_inference_steps=1, guidance_scale=1.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
        except _Stop:
            pass
        finally:
            h.remove()
        return cap["k"]

    def encode_img(np_uint8):
        x = torch.from_numpy(np_uint8).permute(2, 0, 1)[None, :, None].float()  # (1,3,1,H,W)
        x = (x / 127.5 - 1.0).to("cuda", torch.bfloat16)
        with torch.no_grad():
            return pipe._encode_video(x)  # (1,48,1,64,64) normalized

    # build per-pair tensors
    data = []
    for sname, fnum in PAIRS:
        seq = load_sequence(f"{VOL}/dancetrack/train1/{sname}")
        w, h = seq.image_size
        frame = _resize_frame(seq.frame_path(fnum), SIZE)
        arr = load_instance_map(f"{PSEUDO}/{sname}/{fnum:08d}.png")
        idmap = idmap_from_instance_map(resize_instance_map(arr, SIZE), palette, K=48)
        k = capture_i2v(frame)
        rgb_lat = k["vision_tokens"][0][:, :, 0:1].clone()       # clean RGB condition (frame 0)
        idmap_lat = encode_img(idmap)                            # target latent (frame 1)
        n_noisy = int(k["vision_mse_loss_indexes"].numel())
        sx, sy = SIZE / w, SIZE / h
        gt = [type(b)(b.frame, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis)
              for b in seq.boxes_at(fnum)]
        data.append({"name": f"{sname}_f{fnum}", "frame": frame, "idmap": idmap, "k": k,
                     "rgb_lat": rgb_lat, "idmap_lat": idmap_lat, "n_noisy": n_noisy, "gt": gt})
    out["latent_shape"] = tuple(data[0]["k"]["vision_tokens"][0].shape)
    out["n_noisy_tokens"] = data[0]["n_noisy"]
    out["n_total_vision"] = int(data[0]["k"]["vision_sequence_indexes"].numel())
    log(f"latent shape {out['latent_shape']} | noisy {out['n_noisy_tokens']} "
        f"/ total {out['n_total_vision']} (frame0=cond expected)")

    opt = torch.optim.AdamW([p for p in tf.parameters() if p.requires_grad], lr=LR)
    tf.train()
    losses = []
    for step in range(STEPS):
        d = data[step % len(data)]
        idx = int(torch.randint(0, len(timesteps), (1,)))
        t, sigma = timesteps[idx], sigmas[idx].to(torch.bfloat16)
        noise = torch.randn_like(d["idmap_lat"])
        x_t = (1.0 - sigma) * d["idmap_lat"] + sigma * noise
        lat = torch.cat([d["rgb_lat"], x_t], dim=2).to(torch.bfloat16)
        kk = dict(d["k"])
        kk["vision_tokens"] = [lat]
        kk["vision_timesteps"] = torch.full((d["n_noisy"],), float(t), device="cuda", dtype=torch.bfloat16)
        pred = tf(**kk)[0]
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred  # (1,48,2,64,64)
        pred_tgt = pred[:, :, -1:]                                   # target = last latent frame
        vtarget = (noise - d["idmap_lat"])                          # (1,48,1,64,64)
        loss = F.mse_loss(pred_tgt.float(), vtarget.reshape(pred_tgt.shape).float())
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
        if step % 25 == 0:
            log(f"step {step} loss {np.mean(losses[-25:]):.4f}")

    # inference: condition on each frame, take a generated (non-condition) frame as the ID-map
    tf.eval()
    gens = []
    for d in data:
        with torch.no_grad():
            o = pipe(prompt=PROMPT, image=Image.fromarray(d["frame"]), num_frames=NUM_FRAMES,
                     height=SIZE, width=SIZE, num_inference_steps=20, guidance_scale=4.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
        vid = o.video
        gen = np.asarray(vid[-1])           # last frame = generated ID-map (frame0 is the RGB cond)
        gens.append(gen)
        Image.fromarray(d["frame"]).save(f"{RES}/{d['name']}_frame.png")
        Image.fromarray(d["idmap"]).save(f"{RES}/{d['name']}_gt.png")
        Image.fromarray(gen).save(f"{RES}/{d['name']}_gen.png")
        Image.fromarray(np.asarray(vid[0])).save(f"{RES}/{d['name']}_gen_cond0.png")

    # metrics: self-IoU (gen_i vs GT_i) and cross-IoU (gen_0 vs GT_1) — conditioning works
    # iff self >> cross and the two gens differ.
    res = {}
    for i, d in enumerate(data):
        pb = idmaps_to_boxes([gens[i]], palette=palette, min_area=300)
        res[d["name"]] = {
            "self_mean_iou": round(mean_iou(pb, d["gt"]) or 0.0, 3),
            "pred_tracks": len({b.track_id for b in pb}),
            "gt_tracks": len({b.track_id for b in d["gt"]}),
        }
    # cross: gen of pair0 scored against pair1 GT and vice versa
    pb0 = idmaps_to_boxes([gens[0]], palette=palette, min_area=300)
    pb1 = idmaps_to_boxes([gens[1]], palette=palette, min_area=300)
    res["cross_gen0_vs_gt1"] = round(mean_iou(pb0, data[1]["gt"]) or 0.0, 3)
    res["cross_gen1_vs_gt0"] = round(mean_iou(pb1, data[0]["gt"]) or 0.0, 3)
    res["gens_differ_l1"] = float(np.mean(np.abs(gens[0].astype(int) - gens[1].astype(int))))
    out["results"] = res
    out["status"] = "OK"
    log(f"RESULTS: {res}")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-2000:]
    log("COND-SANITY FAILED:\n" + out["error"])

with open(f"{RES}/sanity.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("COND-SANITY-DONE -> " + RES)
