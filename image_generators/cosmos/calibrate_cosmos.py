# Databricks notebook source
# Cosmos3-Super-T2I calibration: load on 2x A100, measure fit + inference speed +
# a REAL LoRA train-step time (via transformer inputs captured during a gen), and
# WRITE everything to the volume (notebook stdout isn't reliably retrievable).

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


pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "transformers",
    "peft", "safetensors", "sentencepiece")

import torch  # noqa: E402

MODEL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/models/cosmos3_super_t2i"
RES = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/results/calibration"
os.makedirs(RES, exist_ok=True)
res = {}


def save():
    with open(os.path.join(RES, "cal.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)


def vram(peak=False):
    f = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return [round(f(i) / 1e9, 1) for i in range(torch.cuda.device_count())]


res["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
res["gpu_total_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 1e9) for i in range(torch.cuda.device_count())]
save()

# ---- load ----
from collections import Counter  # noqa: E402

from diffusers import Cosmos3OmniPipeline  # noqa: E402

try:
    from diffusers import Cosmos3OmniTransformer
except ImportError:
    from diffusers.models import Cosmos3OmniTransformer  # type: ignore

# Pipeline-level device_map treats the 128 GB transformer as ONE unit → can't fit a
# single 85 GB A100 → dumps it on CPU. Instead shard the TRANSFORMER's layers across
# both GPUs (model-level device_map="auto"), then build the pipeline around it.
n_gpu = torch.cuda.device_count()
max_memory = {i: "76GiB" for i in range(n_gpu)}
try:
    t0 = time.time()
    transformer = Cosmos3OmniTransformer.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=torch.bfloat16,
        device_map="auto", max_memory=max_memory)
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, transformer=transformer, torch_dtype=torch.bfloat16,
        enable_safety_checker=False)
    # place the small non-transformer components on GPU 0
    for name in ("vae", "text_encoder", "vision_encoder", "sound_tokenizer"):
        comp = getattr(pipe, name, None)
        if comp is not None and hasattr(comp, "to"):
            try:
                comp.to("cuda:0")
            except Exception:
                pass
    res["load_sec"] = round(time.time() - t0)
    res["vram_after_load_gb"] = vram()
    log(f"loaded in {res['load_sec']}s | VRAM {res['vram_after_load_gb']}")
except Exception:
    res["load_error"] = traceback.format_exc()
    save()
    raise

# ---- introspect transformer ----
tf = pipe.transformer
res["transformer_class"] = type(tf).__name__
res["transformer_params_B"] = round(sum(p.numel() for p in tf.parameters()) / 1e9, 2)
res["linear_leaf_names"] = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
dm = getattr(tf, "hf_device_map", None)
if dm:
    res["transformer_device_spread"] = dict(Counter(str(v) for v in dm.values()))
save()
log(f"transformer {res['transformer_class']} {res['transformer_params_B']}B | linears {res['linear_leaf_names']}")

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
    t0 = time.time()
    steps = 12
    out = pipe(prompt=prompt, num_frames=1, height=1024, width=1024,
               num_inference_steps=steps, guidance_scale=4.0,
               generator=torch.Generator(device="cuda").manual_seed(0))
    dt = time.time() - t0
    res["t2i_total_sec"] = round(dt, 1)
    res["t2i_sec_per_step"] = round(dt / steps, 3)
    res["t2i_peak_vram_gb"] = vram(peak=True)
    out.video[0].save("/tmp/cal_t2i.png")
    log(f"T2I 20steps@1024 {dt:.1f}s ({dt/20:.2f}s/step) peak {res['t2i_peak_vram_gb']}")
except Exception:
    res["t2i_error"] = traceback.format_exc()
finally:
    h.remove()
save()

# ---- REAL LoRA train-step timing using captured transformer inputs ----
try:
    from peft import LoraConfig
    targets = [n for n in res["linear_leaf_names"] if n not in ("proj_out",)] or res["linear_leaf_names"]
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0))
    if hasattr(tf, "enable_gradient_checkpointing"):
        tf.enable_gradient_checkpointing()
    tf.train()
    trainable = [p for p in tf.parameters() if p.requires_grad]
    res["lora_trainable_M"] = round(sum(p.numel() for p in trainable) / 1e6, 2)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    args, kwargs = captured.get("args", ()), captured.get("kwargs", {})
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for it in range(6):
        torch.cuda.synchronize()
        t0 = time.time()
        o = tf(*args, **kwargs)
        sample = o.sample if hasattr(o, "sample") else (o[0] if isinstance(o, (tuple, list)) else o)
        loss = sample.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        step_times.append(round(time.time() - t0, 3))
    res["lora_step_times_sec"] = step_times
    res["lora_step_median_sec"] = sorted(step_times)[len(step_times) // 2]
    res["lora_peak_vram_gb"] = vram(peak=True)
    log(f"LoRA step times {step_times} | peak {res['lora_peak_vram_gb']}")
except Exception:
    res["lora_error"] = traceback.format_exc()
save()

# ---- image-conditioned gen test (num_frames=2: anchor frame0=image, gen frame1) ----
try:
    import numpy as np
    from PIL import Image
    frame = Image.fromarray((np.random.rand(1024, 1024, 3) * 255).astype("uint8"))
    t0 = time.time()
    out = pipe(prompt="segmentation map", image=frame, num_frames=2, height=1024, width=1024,
               num_inference_steps=10, guidance_scale=4.0,
               generator=torch.Generator(device="cuda").manual_seed(0))
    res["img_cond_ok"] = True
    res["img_cond_sec"] = round(time.time() - t0, 1)
    res["img_cond_num_out_frames"] = len(out.video)
    log(f"image-conditioned gen OK: {len(out.video)} frames in {res['img_cond_sec']}s")
except Exception:
    res["img_cond_error"] = traceback.format_exc()
save()

log("CALIBRATION-DONE -> " + os.path.join(RES, "cal.json"))
