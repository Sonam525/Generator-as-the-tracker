# Databricks notebook source
# Phase-1 IC-LoRA on the A100: set up the official Lightricks ltx-trainer, preprocess our aligned
# (RGB, ID-map) manifest into latents, and train the v2v IC-LoRA (RGB clip -> ID-map clip).
# Self-contained for a fresh job cluster. Preprocessing output persists on the volume, so a
# train-only re-run skips it. Native in-context conditioning = the replacement for the abandoned
# hand-rolled cond_sanity.
#
# Stack note: the trainer pins its own deps (incl. a CUDA torch) -> we use an isolated `uv` venv
# (`uv sync`) and invoke the trainer via `uv run`, independent of the Databricks runtime's torch.

import os
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sh(cmd, cwd=None, check=True):
    # Capture then write ONCE per command (FUSE volumes reject the seek that streaming/line-
    # buffering needs -> "Illegal seek"). The notebook API hides subprocess output, so we publish
    # to the volume log. On the warm cluster train.py fails cleanly (rc=1) -> full output captured.
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out, err = (r.stdout or "")[-20000:], (r.stderr or "")[-20000:]
    try:
        with open(LOGFILE, "a") as f:
            f.write(f"\n\n===== $ {cmd}\n[rc={r.returncode}]\nSTDOUT:\n{out}\nSTDERR:\n{err}\n")
    except Exception:
        pass
    if check and r.returncode != 0:
        # embed the output tail IN the exception -> the notebook API's error_trace returns it,
        # independent of the FUSE log or the cluster surviving (robust to hard crashes).
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n"
                           f"--- STDOUT tail ---\n{out[-4000:]}\n--- STDERR tail ---\n{err[-4000:]}")
    return r


VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/ltx2_native/model/ltx-2.3-22b-dev.safetensors"
GEMMA = f"{VOL}/models/ltx2_native/gemma"
DS = f"{VOL}/ltx_ds/phase1"
MANIFEST = f"{DS}/metadata.jsonl"
PRECOMP = f"{DS}/.precomputed"
CONFIG = os.environ["GENTRACK_WS"] + "/video_generator/configs/ltx_v2v_ic_lora_phase1.yaml"
REPO = "/local_disk0/LTX-2"
LOGDIR = f"{VOL}/results/ltx_train_phase1"
os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = f"{LOGDIR}/run.log"
open(LOGFILE, "w").close()  # fresh log each run (readable from the volume)
os.environ["PYTHONUNBUFFERED"] = "1"  # stream train.py output live to the log

# fail fast: every input asset must exist before we spend setup time
for p in (MODEL, GEMMA, MANIFEST, CONFIG):
    assert os.path.exists(p), f"MISSING required input: {p}"
log("inputs present (model, gemma, manifest, config)")

try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    pass

# ---- setup: clone + uv sync (isolated venv with the trainer's pinned deps) ----
if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 https://github.com/Lightricks/LTX-2.git {REPO}")
sh("pip install -q uv", check=False)
sh("uv sync --frozen", cwd=REPO)
log("trainer env ready")

# ---- preprocess: (RGB, ID-map) videos -> latents/ reference_latents/ conditions/ (skip if done) ----
if not os.path.isdir(f"{PRECOMP}/latents"):
    sh(f'uv run python packages/ltx-trainer/scripts/process_dataset.py "{MANIFEST}" '
       f'--resolution-buckets "512x512x49" --model-path "{MODEL}" '
       f'--text-encoder-path "{GEMMA}" --skip-audio --output-dir "{PRECOMP}"', cwd=REPO)
    log("preprocess done")
else:
    log("latents already present — skipping preprocess")

# ---- train the IC-LoRA (output on local disk; published to the volume below) ----
try:
    sh(f'uv run python packages/ltx-trainer/scripts/train.py "{CONFIG}"', cwd=REPO)
    log("PHASE1-TRAIN-DONE")
finally:
    # publish checkpoints + validation sample videos to the volume (plain copy — FUSE-safe),
    # even on failure so partial checkpoints/samples are inspectable.
    OUT_VOL = f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase1"
    sh(f'mkdir -p "{OUT_VOL}" && cp -r /local_disk0/ltx_out/v2v_ic_lora_phase1/. "{OUT_VOL}/" 2>/dev/null || true',
       check=False)
    log(f"published outputs -> {OUT_VOL}")
