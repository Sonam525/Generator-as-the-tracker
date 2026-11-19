# Databricks notebook source
# Injection sanity (P2 step 4b): validate the embedding-bank injection BEFORE Stage-3.
# Mechanism: pad the prompt with filler so the last K_max text tokens are "bank slots"; a
# forward_hook on embed_tokens overwrites them with bank_proj(appearance_i)+color_embed(c_i)
# (null for empty slots). Vision attends to them via the native two-way attention — no index
# /mRoPE surgery. DECISIVE test: overfit ONE frame but RANDOMIZE the appearance->colour
# assignment every step (target ID-map recoloured to match), so the model cannot memorize —
# it must READ the bank to colour correctly. Then held-out assignments + a colour-swap check.

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

from gentrack.color_codec import encode  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.pseudo_masks import load_instance_map, masks_from_instance_map, resize_instance_map  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402
from gentrack.train.stage1_data import _resize_frame  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/cosmos3_nano"
PSEUDO = f"{VOL}/pseudo_masks/dancetrack"
CUE = "pos"   # "pos" = easy spatial cue (mechanism test) | "app" = pooled-appearance (real bank)
RES = f"{VOL}/results/inject_sanity_{CUE}"
os.makedirs(RES, exist_ok=True)
SIZE, NUM_FRAMES, STEPS, LR, K_MAX, APP_DIM = 1024, 5, 800, 2e-4, 47, 48
BASE_PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"
FILLER = " " + " ".join(str(i) for i in range(60))   # >= K_MAX filler tokens at the prompt end
PROMPT = BASE_PROMPT + FILLER
SEQ, FRAME = "dancetrack0008", 1
palette = build(48)

from diffusers import Cosmos3OmniPipeline  # noqa: E402
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler  # noqa: E402
from peft import LoraConfig  # noqa: E402

out = {}
try:
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
    tf = pipe.transformer
    lin = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
    tgt = [n for n in ("to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj") if n in lin]
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=tgt, lora_dropout=0.0))
    if hasattr(tf, "enable_gradient_checkpointing"):
        tf.enable_gradient_checkpointing()
    H = tf.config.hidden_size

    # trainable injection modules (bf16, on cuda)
    bank_proj = torch.nn.Linear(APP_DIM, H).to("cuda", torch.bfloat16)
    pos_proj = torch.nn.Linear(4, H).to("cuda", torch.bfloat16)   # box (cx,cy,w,h) -> token
    color_embed = torch.nn.Embedding(48, H).to("cuda", torch.bfloat16)
    null_tok = torch.nn.Parameter(torch.zeros(H, device="cuda", dtype=torch.bfloat16))
    for m in (bank_proj, pos_proj):
        torch.nn.init.normal_(m.weight, std=0.02)
        torch.nn.init.zeros_(m.bias)
    torch.nn.init.normal_(color_embed.weight, std=0.02)

    tf._bank = None  # (K_MAX, H) set before each forward; hook writes it into the slot positions

    def embed_hook(module, inp, output):
        b = getattr(tf, "_bank", None)
        if b is not None and output.shape[0] >= K_MAX:
            output = output.clone()
            output[-K_MAX:] = b
        return output

    tf.embed_tokens.register_forward_hook(embed_hook)

    sched = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    sched.set_timesteps(1000, device="cuda")
    timesteps, sigmas = sched.timesteps, sched.sigmas

    def capture(frame_np):
        cap = {}

        class _Stop(Exception):
            pass

        def hook(m, a, k):
            cap["k"] = dict(k)
            raise _Stop()

        tf._bank = None
        h = tf.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                pipe(prompt=PROMPT, image=Image.fromarray(frame_np), num_frames=NUM_FRAMES,
                     height=SIZE, width=SIZE, num_inference_steps=1, guidance_scale=1.0,
                     use_system_prompt=True, add_resolution_template=False, add_duration_template=False,
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

    # frame + per-track masks (downscaled to the 64x64 latent grid for appearance pooling)
    seq = load_sequence(f"{VOL}/dancetrack/train1/{SEQ}")
    frame = _resize_frame(seq.frame_path(FRAME), SIZE)
    inst = resize_instance_map(load_instance_map(f"{PSEUDO}/{SEQ}/{FRAME:08d}.png"), SIZE)
    masks_full, tids = masks_from_instance_map(inst)
    tids = tids[:K_MAX]
    masks_full = masks_full[:tids.__len__()]
    out["n_tracks"] = len(tids)

    k = capture(frame)
    frame_lat = k["vision_tokens"][0][:, :, 0:1]            # (1,48,1,64,64) clean RGB condition
    und_text_len = int(k["input_ids"].shape[0])
    out["und_text_len"] = und_text_len
    out["n_noisy"] = int(k["vision_mse_loss_indexes"].numel())
    assert und_text_len >= K_MAX, "filler too short for K_MAX slots"

    # appearance embedding per track: masked mean of the 48-ch condition latent over the region
    lat = frame_lat[0, :, 0]                                 # (48,64,64)
    app = []
    for m in masks_full:
        m64 = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((64, 64), Image.NEAREST)).astype(bool)
        if m64.sum() == 0:
            app.append(torch.zeros(APP_DIM, device="cuda", dtype=torch.bfloat16))
        else:
            sel = torch.from_numpy(m64).to("cuda")
            app.append(lat[:, sel].mean(dim=1))
    app = torch.stack(app)                                   # (N,48) stable appearance, color-independent
    n = len(tids)

    boxfeat = []                                             # (N,4) normalized box (cx,cy,w,h) per track
    for m in masks_full:
        ys, xs = np.where(m)
        boxfeat.append([(xs.min() + xs.max()) / 2 / SIZE, (ys.min() + ys.max()) / 2 / SIZE,
                        (xs.max() - xs.min() + 1) / SIZE, (ys.max() - ys.min() + 1) / SIZE])
    boxfeat = torch.tensor(boxfeat, device="cuda", dtype=torch.bfloat16)

    def build_bank(colors):
        toks = null_tok.unsqueeze(0).expand(K_MAX, H).clone()
        cue = pos_proj(boxfeat) if CUE == "pos" else bank_proj(app)
        active = cue + color_embed(torch.tensor(colors, device="cuda"))
        toks[:n] = active
        return toks

    def render(colors):  # target ID-map with this track->palette-idx assignment
        return encode(masks_full, list(range(n)),
                      palette=palette, assignment={i: colors[i] for i in range(n)})

    params = [*bank_proj.parameters(), *pos_proj.parameters(), *color_embed.parameters(), null_tok,
              *[p for p in tf.parameters() if p.requires_grad]]
    opt = torch.optim.AdamW(params, lr=LR)
    tf.train()
    losses = []
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        colors = list(rng.choice(range(1, 48), size=n, replace=False))  # RANDOM assignment each step
        tf._bank = build_bank(colors)
        x0 = encode_idmap(render(colors))
        noise = torch.randn_like(x0)
        idx = int(torch.randint(0, len(timesteps), (1,)))
        t, sigma = timesteps[idx], sigmas[idx].to(x0.dtype)
        x_t = (1.0 - sigma) * x0 + sigma * noise
        kk = dict(k)
        kk["vision_tokens"] = [torch.cat([frame_lat, x_t], dim=2).to(torch.bfloat16)]
        kk["vision_timesteps"] = torch.full((out["n_noisy"],), float(t), device="cuda", dtype=torch.bfloat16)
        pred = tf(**kk)[0]
        pred = pred[0] if isinstance(pred, (list, tuple)) else pred
        loss = F.mse_loss(pred[:, :, -1:].float(), (noise - x0).reshape(pred[:, :, -1:].shape).float())
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
        if step % 25 == 0:
            log(f"step {step} loss {np.mean(losses[-25:]):.4f}")

    # eval: held-out assignments — does each track get its ASSIGNED colour? (bank binding)
    def color_binding_acc(colors):
        tf.eval()
        tf._bank = build_bank(colors)
        with torch.no_grad():
            o = pipe(prompt=PROMPT, image=Image.fromarray(frame), num_frames=NUM_FRAMES, height=SIZE,
                     width=SIZE, num_inference_steps=20, guidance_scale=1.0, use_system_prompt=True,
                     add_resolution_template=False, add_duration_template=False,
                     generator=torch.Generator(device="cuda").manual_seed(0))
        gen = np.asarray(o.video[-1])
        boxes = idmaps_to_boxes([gen], palette=palette, min_area=300)
        # match each track's region (by IoU of its mask bbox) to a predicted box, check colour idx
        hit = 0
        for i, m in enumerate(masks_full):
            ys, xs = np.where(m)
            gb = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            best, bid = 0.3, None
            for b in boxes:
                bx, by, bw, bh = b.xywh
                ix0, iy0 = max(gb[0], bx), max(gb[1], by)
                ix1, iy1 = min(gb[0] + gb[2], bx + bw), min(gb[1] + gb[3], by + bh)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                u = gb[2] * gb[3] + bw * bh - inter
                iou = inter / u if u > 0 else 0
                if iou > best:
                    best, bid = iou, b.track_id
            if bid == colors[i]:
                hit += 1
        tf.train()
        return round(hit / max(1, n), 3), gen

    rng2 = np.random.default_rng(123)
    a1 = list(rng2.choice(range(1, 48), size=n, replace=False))
    a2 = a1.copy()
    a2[0], a2[1] = a2[1], a2[0]   # swap first two tracks' colours
    acc1, gen1 = color_binding_acc(a1)
    acc2, gen2 = color_binding_acc(a2)
    Image.fromarray(frame).save(f"{RES}/frame.png")
    Image.fromarray(gen1).save(f"{RES}/gen_assign1.png")
    Image.fromarray(gen2).save(f"{RES}/gen_swap.png")
    out["color_binding_acc_assign1"] = acc1
    out["color_binding_acc_swap"] = acc2
    out["swap_changed_output_l1"] = float(np.mean(np.abs(gen1.astype(int) - gen2.astype(int))))
    out["status"] = "OK"
    log(f"RESULTS: binding acc {acc1}/{acc2}, swap L1 {out['swap_changed_output_l1']:.2f}")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-2500:]
    log("INJECT-SANITY FAILED:\n" + out["error"])

with open(f"{RES}/sanity.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("INJECT-SANITY-DONE -> " + RES)
