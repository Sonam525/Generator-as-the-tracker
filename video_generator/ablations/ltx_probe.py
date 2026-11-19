# Databricks notebook source
# LTX-2.3 (22B) calibration + interface probe — the FIRST GPU step of the P2-C video arm,
# the analog of calibrate_cosmos.py. Gates feasibility and reveals the conditioning
# interface BEFORE any training, so cond_mode (channel_concat vs temporal_pack) can be
# decided correctly rather than guessed (parameters are pinned, never invented).
#
# Stages (each wrapped + saved incrementally so a late OOM/error still yields the earlier
# facts):
#   A. load 22B on 1xA100, VRAM after load                          -> does it even fit?
#   B. introspect: transformer params, Linear names (LoRA targets), __call__ signature
#      (audio discovery), VAE ratio attributes
#   C. VAE compression EMPIRICAL: encode a real-size clip, read latent shape -> confirm
#      temporal_ratio (flagged 8 in video_generator/configs/ltx_vid.yaml) + spatial ratio + in_channels
#   D. conditioning interface: run a real RGB-clip-conditioned gen, CAPTURE the transformer
#      forward kwargs (names/shapes/dtypes) -> reveals how conditioned tokens are marked,
#      which both cond modes need; also inference sec/step + peak VRAM
#   E. LoRA forward/backward at the real (49f, 512) shapes -> THE 22B-video-fit gate
#
# Output: results/ltx_probe/probe.json. Read it back, then write ltx_cond_sanity.py (the
# two-mode self-IoU>>cross decision) against the captured interface — same calibrate->decide
# two-step the Cosmos arm used (calibrate_cosmos -> cond_sanity).

import inspect
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
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# LTX-2 pipelines need a recent diffusers (the ltx2 module lives on main).
pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "transformers",
    "peft", "safetensors", "sentencepiece", "imageio")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gentrack.models.ltx import LORA_TARGETS, latent_frames  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/ltx2_3"     # 22B dev weights (human: download to this dot-free dir)
RES = f"{VOL}/results/ltx_probe"
os.makedirs(RES, exist_ok=True)

SIZE = 512          # video_generator/configs/ltx_vid.yaml resolution (512 % 32 == 0)
NUM_FRAMES = 49     # video_generator/configs/ltx_vid.yaml window_frames (49 = 8*6+1)
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"

res = {"size": SIZE, "num_frames": NUM_FRAMES, "expected_latent_frames_if_ratio8": latent_frames(NUM_FRAMES, 8)}


def save():
    with open(f"{RES}/probe.json", "w") as f:
        json.dump(res, f, indent=2, default=str)


def vram(peak=False):
    g = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return [round(g(i) / 1e9, 1) for i in range(torch.cuda.device_count())]


def desc(v):
    """Compact description of a captured forward arg (for the interface dump)."""
    if torch.is_tensor(v):
        return {"shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return [desc(x) for x in v][:8]
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return f"{type(v).__name__}"


res["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
res["gpu_total_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 1e9)
                       for i in range(torch.cuda.device_count())]
save()

# ---- A. load ----
# MEASURED FINDING (download component sizes): full-pipeline weights ~101 GB bf16 —
# transformer 38, text_encoder ~49 (a ~24B LLM), connectors ~13, vae ~1.5. Weights ALONE
# exceed an 80 GB A100, so naive .to("cuda") OOMs (§6.4 doesn't-fit finding). Load via
# model_cpu_offload (host ~220 GB RAM) — keeps only the active component resident, still
# exercises every component so stages B-E extract the VAE ratio + conditioning interface.
# The REAL training path drops the 49 GB text encoder via precomputed prompt embeds (the
# prompt is fixed) -> VAE+DiT ~40 GB fits one A100; that path is designed from this probe.
from diffusers import LTX2ConditionPipeline  # noqa: E402

res["naive_full_load_fits_80gb"] = False
res["full_pipeline_weights_gb"] = 101
res["load_mode"] = "cpu_offload"
try:
    t0 = time.time()
    pipe = LTX2ConditionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    res["load_sec"] = round(time.time() - t0)
    res["vram_after_load_gb"] = vram()
    log(f"loaded 22B (cpu_offload) in {res['load_sec']}s | resident VRAM "
        f"{res['vram_after_load_gb']} / {res['gpu_total_gb']}")
except Exception:
    res["load_error"] = traceback.format_exc()[-3000:]
    save()
    log("LOAD FAILED even with cpu_offload:\n" + res["load_error"])
    raise

# ---- B. introspect ----
try:
    tf = pipe.transformer
    res["transformer_class"] = type(tf).__name__
    res["transformer_params_B"] = round(sum(p.numel() for p in tf.parameters()) / 1e9, 2)
    res["linear_leaf_names"] = sorted({n.split(".")[-1] for n, m in tf.named_modules()
                                       if isinstance(m, torch.nn.Linear)})
    res["lora_targets_matched"] = [n for n in LORA_TARGETS if n in res["linear_leaf_names"]]
    cfg = getattr(tf, "config", None)
    if cfg is not None:
        res["transformer_config"] = {k: getattr(cfg, k) for k in
            ("in_channels", "out_channels", "patch_size", "patch_size_t", "num_layers",
             "caption_channels", "num_attention_heads", "attention_head_dim")
            if hasattr(cfg, k)}
    res["call_params"] = list(inspect.signature(pipe.__call__).parameters)
    res["audio_params"] = [p for p in res["call_params"] if "audio" in p.lower()]
    res["pipe_components"] = [n for n in vars(pipe) if not n.startswith("_")]
    for attr in ("vae_spatial_compression_ratio", "vae_temporal_compression_ratio"):
        if hasattr(pipe, attr):
            res[attr] = getattr(pipe, attr)
    save()
    log(f"transformer {res['transformer_class']} {res['transformer_params_B']}B | "
        f"in_ch {res.get('transformer_config', {}).get('in_channels')} | "
        f"lora targets {res['lora_targets_matched']} | audio params {res['audio_params']}")
except Exception:
    res["introspect_error"] = traceback.format_exc()[-3000:]
    save()

# A throwaway RGB "clip" reused by stages C (VAE) and D (conditioning) so D survives a C failure.
clip = (np.random.rand(NUM_FRAMES, SIZE, SIZE, 3) * 255).astype(np.uint8)

# ---- C. VAE compression, empirical ----
try:
    x = torch.from_numpy(clip).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0  # (1,3,F,H,W)
    x = x.to("cuda", torch.bfloat16)
    with torch.no_grad():
        enc = pipe.vae.encode(x)
    if hasattr(enc, "latent_dist"):
        lat = enc.latent_dist.sample()
    elif hasattr(enc, "latent"):
        lat = enc.latent
    else:
        lat = enc[0]
    res["vae_latent_shape"] = list(lat.shape)         # (1, C, T, H, W)
    _, c, t, hh, ww = lat.shape
    res["vae_measured"] = {
        "latent_channels": c,
        "spatial_ratio": round(SIZE / hh, 2),
        "temporal_ratio_causal": round((NUM_FRAMES - 1) / (t - 1), 2) if t > 1 else None,
        "temporal_ratio_naive": round(NUM_FRAMES / t, 2),
        "latent_frames": t,
    }
    save()
    log(f"VAE latent {res['vae_latent_shape']} -> spatial {res['vae_measured']['spatial_ratio']}x, "
        f"temporal(causal) {res['vae_measured']['temporal_ratio_causal']}x, {t} latent frames "
        f"(config assumes ratio 8 -> {res['expected_latent_frames_if_ratio8']})")
    del x, enc, lat
    torch.cuda.empty_cache()
except Exception:
    res["vae_error"] = traceback.format_exc()[-3000:]
    save()

# ---- D. conditioning interface: capture transformer kwargs during an RGB-clip-conditioned gen ----
captured = {}


def _hook(module, args, kwargs):
    if "kwargs" not in captured:
        captured["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
        captured["kwargs"] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}


try:
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    rgb_clip = [Image.fromarray(clip[i]) for i in range(NUM_FRAMES)]
    cond = LTX2VideoCondition(frames=rgb_clip, index=0, strength=1.0)
    h = tf.register_forward_pre_hook(_hook, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        gen_out = pipe(conditions=[cond], prompt=PROMPT, num_frames=NUM_FRAMES, height=SIZE,
             width=SIZE, num_inference_steps=8, guidance_scale=3.0, output_type="latent",
             return_dict=False, generator=torch.Generator(device="cuda").manual_seed(0))
    dt = time.time() - t0
    res["cond_gen_sec_8steps"] = round(dt, 1)
    res["cond_gen_sec_per_step"] = round(dt / 8, 2)
    res["cond_gen_peak_vram_gb"] = vram(peak=True)
    res["gen_latent_out_desc"] = desc(gen_out)   # latent temporal dim -> VAE ratio cross-check
    res["transformer_forward_args"] = [desc(a) for a in captured.get("args", ())]
    res["transformer_forward_kwargs"] = {k: desc(v) for k, v in captured.get("kwargs", {}).items()}
    h.remove()
    save()
    log(f"conditioned gen {dt:.1f}s ({res['cond_gen_sec_per_step']}s/step) peak "
        f"{res['cond_gen_peak_vram_gb']} | fwd kwargs {list(res['transformer_forward_kwargs'])}")
except Exception:
    res["cond_interface_error"] = traceback.format_exc()[-3000:]
    try:
        h.remove()
    except Exception:
        pass
    save()

# safety net: persist the captured interface even if pipe() raised AFTER the hook fired
# (the interface dump is what ltx_cond_sanity.py is written against — must not be lost).
if captured.get("kwargs") and "transformer_forward_kwargs" not in res:
    res["transformer_forward_args"] = [desc(a) for a in captured.get("args", ())]
    res["transformer_forward_kwargs"] = {k: desc(v) for k, v in captured["kwargs"].items()}
    res["cond_interface_recovered_from_hook"] = True
    save()
    log("recovered transformer interface from hook despite a stage-D error")

# ---- E. LoRA forward/backward fit (THE 22B-video-fit gate) using captured kwargs ----
# Measure the REAL training footprint: the precompute-embeds path loads only VAE+DiT, so pin
# the transformer SOLELY on the GPU (drop the cpu_offload hooks; the other ~62 GB of
# components stay on CPU and are not needed — conditioning is already in the captured kwargs).
# This is the v2v-LoRA fit number that actually gates training, not the offloaded one.
try:
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(tf, recurse=True)
    torch.cuda.empty_cache()
    tf.to("cuda")
    res["lora_fit_context"] = "transformer-only on GPU (precompute-embeds training path)"
    res["vram_transformer_only_gb"] = vram()
    log(f"transformer-only on GPU: {res['vram_transformer_only_gb']} / {res['gpu_total_gb']}")
except Exception:
    res["lora_offload_detach_error"] = traceback.format_exc()[-1500:]

try:
    from peft import LoraConfig
    targets = res.get("lora_targets_matched") or res.get("linear_leaf_names")
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0))
    if hasattr(tf, "enable_gradient_checkpointing"):
        tf.enable_gradient_checkpointing()
    tf.train()
    trainable = [p for p in tf.parameters() if p.requires_grad]
    res["lora_trainable_M"] = round(sum(p.numel() for p in trainable) / 1e6, 2)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    args, kwargs = captured.get("args", ()), captured.get("kwargs", {})
    if not args and not kwargs:
        raise RuntimeError("no captured transformer inputs (stage D failed) — cannot time LoRA")
    torch.cuda.reset_peak_memory_stats()
    step_times = []
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
        step_times.append(round(time.time() - t0, 3))
    res["lora_step_times_sec"] = step_times
    res["lora_step_median_sec"] = sorted(step_times)[len(step_times) // 2]
    res["lora_peak_vram_gb"] = vram(peak=True)
    res["FIT_VERDICT"] = ("FITS" if max(res["lora_peak_vram_gb"]) < min(res["gpu_total_gb"])
                          else "TIGHT/OOM-RISK")
    save()
    log(f"LoRA {res['lora_trainable_M']}M | step {res['lora_step_times_sec']} | "
        f"peak {res['lora_peak_vram_gb']} / {res['gpu_total_gb']} -> {res['FIT_VERDICT']}")
except Exception:
    res["lora_error"] = traceback.format_exc()[-3000:]
    save()

res["status"] = "DONE"
save()
log("LTX-PROBE-DONE -> " + f"{RES}/probe.json")
