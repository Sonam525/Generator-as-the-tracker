# Databricks notebook source
# Download the DIFFUSERS-FORMAT LTX-2.3 (22B) weights to the UC volume for the P2-C video arm.
# Detached job; resumable (HF cache). Dir name has no dots (loader requirement).
#
# Repo MUST be the diffusers-format conversion, NOT the native Lightricks checkpoint:
#   LTX2ConditionPipeline.from_pretrained(local_dir) needs the diffusers layout
#   (model_index.json + subfolder configs). The native `Lightricks/LTX-2.3` repo will NOT load.
#
# HUMAN-IN-THE-LOOP (left to a person, see README.md): LTX-2.3 ships under the Lightricks community licence —
# read it and accept the HF terms on the model page first, or the gated download 401s.
# HF_TOKEN comes from Databricks secrets/env (never printed/logged/committed).

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

try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

REPO = "diffusers/LTX-2.3-Diffusers"   # diffusers-format 22B (verified: from_pretrained target)
DST = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/models/ltx2_3"
os.makedirs(DST, exist_ok=True)

marker = os.path.join(DST, ".download_complete")
if os.path.exists(marker):
    log("already complete (marker present), skipping")
else:
    log(f"downloading {REPO} -> {DST}")
    path = snapshot_download(
        repo_id=REPO, local_dir=DST, max_workers=8,
        token=HF_TOKEN,  # gated: requires accepted licence + a valid token
    )
    open(marker, "w").write("ok")
    log(f"download complete: {path}")

files = glob.glob(DST + "/**/*", recursive=True)
weights = [f for f in files if f.endswith(".safetensors")]
total_gb = sum(os.path.getsize(f) for f in weights) / 1e9
log(f"files={len(files)} safetensors={len(weights)} total={total_gb:.1f} GB")
log("DONE")
