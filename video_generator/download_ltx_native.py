# Databricks notebook source
# Download the NATIVE-format LTX-2.3 assets for the ltx-trainer (IC-LoRA pivot) — NOT the
# diffusers format (that was for the abandoned hand-rolled cond_sanity). The official
# Lightricks/LTX-2 trainer + ltx-pipelines consume native .safetensors + the Gemma text encoder.
#
#   model       : ltx-2.3-22b-dev.safetensors (single file; VAE is bundled in it) ~44 GB
#   text encoder: google/gemma-3-12b-it-qat-q4_0-unquantized (full repo) ~24 GB
#
# Only the ONE model file is fetched (the Lightricks/LTX-2.3 repo also holds distilled +
# upscalers + every IC-LoRA = hundreds of GB — do NOT snapshot the whole repo).
#
# HUMAN-IN-THE-LOOP (left to a person, see README.md): Gemma is GATED under Google's licence — the HF account
# behind HF_TOKEN must have accepted it on the model page, or the download 401s. Lightricks/LTX-2.3
# is under the LTX Open Source licence (the diffusers mirror was ungated).

import glob
import os
import subprocess
import sys
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    _pip("huggingface_hub")
    from huggingface_hub import hf_hub_download, snapshot_download

try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

ROOT = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/models/ltx2_native"
MODEL_DIR = f"{ROOT}/model"
GEMMA_DIR = f"{ROOT}/gemma"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GEMMA_DIR, exist_ok=True)

marker = os.path.join(ROOT, ".download_complete")
if os.path.exists(marker):
    log("already complete (marker present), skipping")
else:
    log("downloading ltx-2.3-22b-dev.safetensors (single file)")
    p = hf_hub_download(repo_id="Lightricks/LTX-2.3", filename="ltx-2.3-22b-dev.safetensors",
                        local_dir=MODEL_DIR, token=HF_TOKEN)
    log(f"  model -> {p}")
    log("downloading google/gemma-3-12b-it-qat-q4_0-unquantized (full repo)")
    g = snapshot_download(repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                          local_dir=GEMMA_DIR, max_workers=8, token=HF_TOKEN)
    log(f"  gemma -> {g}")
    open(marker, "w").write("ok")

files = glob.glob(ROOT + "/**/*", recursive=True)
weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
total_gb = sum(os.path.getsize(f) for f in weights) / 1e9
log(f"weight files={len(weights)} total={total_gb:.1f} GB")
log("DONE")
