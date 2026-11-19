# Databricks notebook source
# Download nvidia/Cosmos3-Super-Text2Image (~123 GB, ungated, bf16) to the UC volume.
# Detached job; resumable (HF cache). Dir name has no dots (loader requirement).

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
    from huggingface_hub import snapshot_download
except ImportError:
    _pip("huggingface_hub")
    from huggingface_hub import snapshot_download

REPO = "nvidia/Cosmos3-Super-Text2Image"
DST = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/models/cosmos3_super_t2i"
os.makedirs(DST, exist_ok=True)

marker = os.path.join(DST, ".download_complete")
if os.path.exists(marker):
    log("already complete (marker present), skipping")
else:
    log(f"downloading {REPO} -> {DST}")
    path = snapshot_download(
        repo_id=REPO, local_dir=DST, max_workers=8,
        token=os.environ.get("HF_TOKEN"),  # ungated; None is fine
    )
    open(marker, "w").write("ok")
    log(f"download complete: {path}")

files = glob.glob(DST + "/**/*", recursive=True)
weights = [f for f in files if f.endswith(".safetensors")]
total_gb = sum(os.path.getsize(f) for f in weights) / 1e9
log(f"files={len(files)} safetensors={len(weights)} total={total_gb:.1f} GB")
log("DONE")
