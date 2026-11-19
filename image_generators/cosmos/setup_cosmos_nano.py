# Databricks notebook source
# Cosmos3-Nano (gated, ~32 GB, omni/image-capable): download to volume + calibrate on
# ONE A100 (no sharding). Measures fit, inference s/step, a real LoRA train-step time,
# and image-conditioned gen. Writes results to the volume (notebook stdout isn't retrievable).

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


# HF token from the Databricks secret scope (gated model)
try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "transformers",
    "peft", "safetensors", "sentencepiece", "huggingface_hub")

import torch  # noqa: E402

REPO = "nvidia/Cosmos3-Nano"
MODEL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/models/cosmos3_nano"
RES = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/results/calibration"
os.makedirs(MODEL, exist_ok=True)
os.makedirs(RES, exist_ok=True)
res = {}


def save():
    with open(os.path.join(RES, "cal_nano.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)


def vram(peak=False):
    f = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return [round(f(i) / 1e9, 1) for i in range(torch.cuda.device_count())]


# ---- download ----
from huggingface_hub import snapshot_download  # noqa: E402

marker = os.path.join(MODEL, ".download_complete")
if not os.path.exists(marker):
    log(f"downloading {REPO} ...")
    t0 = time.time()
    snapshot_download(repo_id=REPO, local_dir=MODEL, max_workers=8, token=HF_TOKEN)
    open(marker, "w").write("ok")
    log(f"downloaded in {time.time()-t0:.0f}s")
res["download_present"] = True
save()

# ---- load on a single GPU ----
from diffusers import Cosmos3OmniPipeline  # noqa: E402

res["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
try:
    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
    res["load_sec"] = round(time.time() - t0)
    res["vram_after_load_gb"] = vram()
    log(f"loaded in {res['load_sec']}s | VRAM {res['vram_after_load_gb']}")
except Exception:
    res["load_error"] = traceback.format_exc()
    save()
    raise

tf = pipe.transformer
res["transformer_class"] = type(tf).__name__
res["transformer_params_B"] = round(sum(p.numel() for p in tf.parameters()) / 1e9, 2)
res["linear_leaf_names"] = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
save()
log(f"transformer {res['transformer_class']} {res['transformer_params_B']}B")

# ---- capture transformer inputs during a sanity gen + time inference ----
captured = {}


def _hook(module, args, kwargs):
    if "args" not in captured:
        captured["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
        captured["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}


h = tf.register_forward_pre_hook(_hook, with_kwargs=True)
try:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    cap = os.path.join(MODEL, "assets", "example_caption.json")
    prompt = json.dumps(json.load(open(cap))) if os.path.exists(cap) else "a red cube on a wooden table"
    torch.cuda.reset_peak_memory_stats()
    steps = 12
    t0 = time.time()
    out = pipe(prompt=prompt, num_frames=1, height=1024, width=1024,
               num_inference_steps=steps, guidance_scale=4.0,
               generator=torch.Generator(device="cuda").manual_seed(0))
    dt = time.time() - t0
    res["t2i_total_sec"] = round(dt, 1)
    res["t2i_sec_per_step"] = round(dt / steps, 3)
    res["t2i_peak_vram_gb"] = vram(peak=True)
    out.video[0].save("/tmp/nano_t2i.png")
    log(f"T2I {steps}steps@1024 {dt:.1f}s ({dt/steps:.2f}s/step) peak {res['t2i_peak_vram_gb']}")
except Exception:
    res["t2i_error"] = traceback.format_exc()
finally:
    h.remove()
save()

# ---- real LoRA train-step timing ----
try:
    from peft import LoraConfig
    targets = [n for n in res["linear_leaf_names"] if n not in ("proj_out", "lm_head")] or res["linear_leaf_names"]
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0))
    if hasattr(tf, "enable_gradient_checkpointing"):
        tf.enable_gradient_checkpointing()
    tf.train()
    trainable = [p for p in tf.parameters() if p.requires_grad]
    res["lora_trainable_M"] = round(sum(p.numel() for p in trainable) / 1e6, 2)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    args, kwargs = captured.get("args", ()), captured.get("kwargs", {})
    torch.cuda.reset_peak_memory_stats()
    st = []
    for _ in range(6):
        torch.cuda.synchronize()
        t0 = time.time()
        o = tf(*args, **kwargs)
        sample = o.sample if hasattr(o, "sample") else (o[0] if isinstance(o, (tuple, list)) else o)
        loss = sample.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        st.append(round(time.time() - t0, 3))
    res["lora_step_times_sec"] = st
    res["lora_step_median_sec"] = sorted(st)[len(st) // 2]
    res["lora_peak_vram_gb"] = vram(peak=True)
    log(f"LoRA step times {st} | peak {res['lora_peak_vram_gb']}")
except Exception:
    res["lora_error"] = traceback.format_exc()
save()

# ---- image-conditioned gen (num_frames=2: anchor frame0=image, gen frame1) ----
try:
    import numpy as np
    from PIL import Image
    frame = Image.fromarray((np.random.rand(1024, 1024, 3) * 255).astype("uint8"))
    t0 = time.time()
    out = pipe(prompt="segmentation map", image=frame, num_frames=2, height=1024, width=1024,
               num_inference_steps=10, guidance_scale=4.0,
               generator=torch.Generator(device="cuda").manual_seed(0))
    res["img_cond_ok"] = True
    res["img_cond_num_out_frames"] = len(out.video)
    log(f"image-conditioned gen OK: {len(out.video)} frames")
except Exception:
    res["img_cond_error"] = traceback.format_exc()
save()

log("NANO-SETUP-DONE -> " + os.path.join(RES, "cal_nano.json"))
