# Databricks notebook source
# Cross-attention bank injection sanity (P2 step 4b, Option A — SAM-faithful). Where the
# text-slot injection was IGNORED (binding 0.0), here the generation tokens get a dedicated
# cross-attention into the bank tokens (K/V) at every decoder layer — zero-initialized out
# so it starts as a no-op and learns. Injected via per-layer forward hooks (no block rewrite),
# grad-checkpointing OFF so hooks fire once. Clean A/B: SAME positional cue that failed via
# text-slots. Overfit 1 frame, RANDOMIZE colour assignment/step (model must read the bank),
# then held-out assignment + colour-swap. If this binds, cross-attn is the mechanism.

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
CUE = "pos"   # mechanism test: same easy spatial cue that failed via text-slots
RES = f"{VOL}/results/xattn_sanity_{CUE}"
os.makedirs(RES, exist_ok=True)
SIZE, NUM_FRAMES, STEPS, LR, XDIM = 1024, 5, 800, 2e-4, 256
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"
SEQ, FRAME = "dancetrack0008", 1
palette = build(48)

from diffusers import Cosmos3OmniPipeline  # noqa: E402
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler  # noqa: E402
from peft import LoraConfig  # noqa: E402


class XAttn(torch.nn.Module):
    """Single-head cross-attention: generation tokens (query) attend to bank tokens (K/V).
    Output projection zero-initialized → the layer starts as a no-op residual and learns."""

    def __init__(self, dim, d):
        super().__init__()
        self.q = torch.nn.Linear(dim, d, bias=False)
        self.k = torch.nn.Linear(dim, d, bias=False)
        self.v = torch.nn.Linear(dim, d, bias=False)
        self.o = torch.nn.Linear(d, dim, bias=False)
        self.scale = d ** -0.5
        torch.nn.init.zeros_(self.o.weight)

    def forward(self, gen, bank):
        q, k, v = self.q(gen), self.k(bank), self.v(bank)
        a = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1) @ v
        return self.o(a)


out = {}
try:
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
    tf = pipe.transformer
    tf.gradient_checkpointing = False
    H = tf.config.hidden_size
    lin = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
    tgt = [n for n in ("to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj") if n in lin]
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=tgt, lora_dropout=0.0))

    # one cross-attn per decoder layer; injected via forward hooks on the layer output (und, gen)
    xattns = torch.nn.ModuleList([XAttn(H, XDIM) for _ in tf.layers]).to("cuda", torch.bfloat16)
    tf._bank = None

    def make_hook(xa):
        def hook(mod, inp, output):
            if tf._bank is not None and isinstance(output, tuple) and len(output) == 2:
                und, gen = output
                return (und, gen + xa(gen, tf._bank))
            return output
        return hook

    for layer, xa in zip(tf.layers, xattns):
        layer.register_forward_hook(make_hook(xa))

    pos_proj = torch.nn.Linear(4, H).to("cuda", torch.bfloat16)
    bank_proj = torch.nn.Linear(48, H).to("cuda", torch.bfloat16)
    color_embed = torch.nn.Embedding(48, H).to("cuda", torch.bfloat16)
    for m in (pos_proj, bank_proj):
        torch.nn.init.normal_(m.weight, std=0.02)
        torch.nn.init.zeros_(m.bias)
    torch.nn.init.normal_(color_embed.weight, std=0.02)

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

    seq = load_sequence(f"{VOL}/dancetrack/train1/{SEQ}")
    frame = _resize_frame(seq.frame_path(FRAME), SIZE)
    inst = resize_instance_map(load_instance_map(f"{PSEUDO}/{SEQ}/{FRAME:08d}.png"), SIZE)
    masks_full, tids = masks_from_instance_map(inst)
    n = len(tids)
    out["n_tracks"] = n

    k = capture(frame)
    frame_lat = k["vision_tokens"][0][:, :, 0:1]
    out["n_noisy"] = int(k["vision_mse_loss_indexes"].numel())

    lat = frame_lat[0, :, 0]
    app, boxfeat = [], []
    for m in masks_full:
        ys, xs = np.where(m)
        boxfeat.append([(xs.min() + xs.max()) / 2 / SIZE, (ys.min() + ys.max()) / 2 / SIZE,
                        (xs.max() - xs.min() + 1) / SIZE, (ys.max() - ys.min() + 1) / SIZE])
        m64 = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((64, 64), Image.NEAREST)).astype(bool)
        sel = torch.from_numpy(m64).to("cuda")
        app.append(lat[:, sel].mean(dim=1) if m64.sum() else torch.zeros(48, device="cuda", dtype=torch.bfloat16))
    app = torch.stack(app)
    boxfeat = torch.tensor(boxfeat, device="cuda", dtype=torch.bfloat16)

    def build_bank(colors):
        cue = pos_proj(boxfeat) if CUE == "pos" else bank_proj(app)
        return cue + color_embed(torch.tensor(colors, device="cuda"))   # (n, H) bank tokens

    def render(colors):
        return encode(masks_full, list(range(n)), palette=palette,
                      assignment={i: colors[i] for i in range(n)})

    params = [*xattns.parameters(), *pos_proj.parameters(), *bank_proj.parameters(),
              *color_embed.parameters(), *[p for p in tf.parameters() if p.requires_grad]]
    opt = torch.optim.AdamW(params, lr=LR)
    tf.train()
    losses = []
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        colors = list(rng.choice(range(1, 48), size=n, replace=False))
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

    def binding_acc(colors):
        tf.eval()
        tf._bank = build_bank(colors)
        with torch.no_grad():
            o = pipe(prompt=PROMPT, image=Image.fromarray(frame), num_frames=NUM_FRAMES, height=SIZE,
                     width=SIZE, num_inference_steps=20, guidance_scale=1.0,
                     generator=torch.Generator(device="cuda").manual_seed(0))
        gen = np.asarray(o.video[-1])
        boxes = idmaps_to_boxes([gen], palette=palette, min_area=300)
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
                if u > 0 and inter / u > best:
                    best, bid = inter / u, b.track_id
            if bid == colors[i]:
                hit += 1
        tf.train()
        return round(hit / max(1, n), 3), gen

    rng2 = np.random.default_rng(123)
    a1 = list(rng2.choice(range(1, 48), size=n, replace=False))
    a2 = a1.copy()
    a2[0], a2[1] = a2[1], a2[0]
    acc1, gen1 = binding_acc(a1)
    acc2, gen2 = binding_acc(a2)
    Image.fromarray(frame).save(f"{RES}/frame.png")
    Image.fromarray(gen1).save(f"{RES}/gen_assign1.png")
    Image.fromarray(gen2).save(f"{RES}/gen_swap.png")
    out["color_binding_acc_assign1"] = acc1
    out["color_binding_acc_swap"] = acc2
    out["swap_changed_output_l1"] = float(np.mean(np.abs(gen1.astype(int) - gen2.astype(int))))
    out["xattn_params_M"] = round(sum(p.numel() for p in xattns.parameters()) / 1e6, 1)
    out["status"] = "OK"
    log(f"RESULTS: binding {acc1}/{acc2}, swap L1 {out['swap_changed_output_l1']:.2f}")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-2500:]
    log("XATTN-SANITY FAILED:\n" + out["error"])

with open(f"{RES}/sanity.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("XATTN-SANITY-DONE -> " + RES)
