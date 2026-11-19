# Databricks notebook source
# Conditioning probe: how does Cosmos3OmniPipeline inject `image=` into the transformer?
# Is the input frame available SPATIALLY (a co-located latent the output can lock to) or
# only as semantic/understanding tokens? This decides the grounding fix (img2img / latent
# concat / native spatial cond). Read-only introspection — no training. Dumps __call__ /
# forward signatures, the captured transformer kwargs (keys + tensor shapes), and the
# __call__ source to the volume (notebook stdout isn't retrievable via CLI).

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

pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "transformers",
    "peft", "safetensors", "sentencepiece")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/cosmos3_nano"
RES = f"{VOL}/results/cond_probe"
os.makedirs(RES, exist_ok=True)
SIZE = 1024
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"


def describe(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]" + (f" of {describe(v[0])}" if v else "")
    if isinstance(v, dict):
        return {str(kk): describe(vv) for kk, vv in v.items()}
    if v is None:
        return "None"
    return type(v).__name__


def src_to(name, obj):
    try:
        s = inspect.getsource(obj)
        with open(f"{RES}/{name}.txt", "w") as f:
            f.write(s)
        return s
    except Exception as e:
        return f"<no source: {e}>"


out = {}
try:
    from diffusers import Cosmos3OmniPipeline

    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
    tf = pipe.transformer

    out["call_params"] = list(inspect.signature(type(pipe).__call__).parameters)
    out["forward_params"] = list(inspect.signature(tf.forward).parameters)
    out["pipe_public_attrs"] = [a for a in dir(pipe) if not a.startswith("_")]
    out["pipe_private_methods"] = [a for a in dir(pipe)
                                   if a.startswith("_") and not a.startswith("__")
                                   and callable(getattr(pipe, a, None))]
    out["has"] = {n: hasattr(pipe, n) for n in
                  ["vae", "image_processor", "prepare_latents", "_encode_video",
                   "encode_image", "_encode_image", "image_encoder", "video_processor"]}

    # capture the kwargs the transformer actually receives for an image-conditioned gen
    cap = {}

    class _Stop(Exception):
        pass

    def hook(m, a, k):
        cap["k"] = dict(k)
        cap["args"] = [describe(x) for x in a]
        raise _Stop()

    frame = Image.fromarray(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    h = tf.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            pipe(prompt=PROMPT, image=frame, num_frames=1, height=SIZE, width=SIZE,
                 num_inference_steps=1, guidance_scale=1.0,
                 generator=torch.Generator(device="cuda").manual_seed(0))
    except _Stop:
        pass
    finally:
        h.remove()
    out["transformer_kwargs"] = {k: describe(v) for k, v in cap.get("k", {}).items()}
    out["transformer_args"] = cap.get("args", [])

    # source dumps for the conditioning path
    src = src_to("call_source", type(pipe).__call__)
    out["image_lines"] = [ln.strip() for ln in src.splitlines() if "image" in ln.lower()][:80]
    for n in ["prepare_latents", "_encode_video", "encode_image", "_encode_image"]:
        if hasattr(pipe, n):
            src_to(f"{n}_source", getattr(pipe, n))
    out["status"] = "OK"
    log("COND-PROBE OK")
except Exception:
    out["status"] = "FAILED"
    out["error"] = traceback.format_exc()[-2000:]
    log("COND-PROBE FAILED:\n" + out["error"])

with open(f"{RES}/probe.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
log("COND-PROBE-DONE -> " + RES)
