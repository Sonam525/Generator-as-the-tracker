# Databricks notebook source
# Download DanceTrack (train + val) from Hugging Face to the UC volume.
# Detached job; resumable (HF cache resume + per-zip done-markers). The test
# split is deferred (submission-server only, working-agreement rule 7, README.md).
#
# Source: https://huggingface.co/datasets/noahcao/dancetrack  (Google Drive deprecated)

import glob
import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    _pip("huggingface_hub")
    from huggingface_hub import hf_hub_download

REPO = "noahcao/dancetrack"
ROOT = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/dancetrack"
STAGE = "/local_disk0/dancetrack_zips"  # fast ephemeral driver SSD
try:
    FILES = [f.strip() for f in dbutils.widgets.get("files").split(",") if f.strip()]  # noqa: F821
except Exception:
    FILES = ["train1.zip", "train2.zip", "val.zip"]  # test1/test2 = end-of-project (rule 7)
TOKEN = os.environ.get("HF_TOKEN")  # public dataset; None = anonymous is fine

os.makedirs(STAGE, exist_ok=True)
os.makedirs(ROOT, exist_ok=True)

for fname in FILES:
    marker = pathlib.Path(ROOT) / f".done_{fname}"
    if marker.exists():
        log(f"{fname}: already extracted (marker present), skipping")
        continue
    log(f"{fname}: downloading from {REPO} ...")
    zp = hf_hub_download(
        repo_id=REPO, filename=fname, repo_type="dataset", local_dir=STAGE, token=TOKEN
    )
    log(f"{fname}: downloaded {os.path.getsize(zp) / 1e9:.2f} GB -> {zp}; extracting ...")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        log(f"{fname}: top entries {names[:3]}")
        z.extractall(ROOT)
    marker.write_text("ok")
    log(f"{fname}: extracted, marker written")

# Normalize a possible 'dancetrack/' top-level prefix into ROOT/{train,val}
nested = pathlib.Path(ROOT) / "dancetrack"
if nested.is_dir():
    log("flattening nested dancetrack/ prefix")
    for sub in nested.iterdir():
        dest = pathlib.Path(ROOT) / sub.name
        if not dest.exists():
            shutil.move(str(sub), str(dest))
    try:
        nested.rmdir()
    except OSError:
        pass

# Verify
for split in ("train", "val", "test"):
    seqs = sorted(glob.glob(f"{ROOT}/{split}/dancetrack*"))
    sample = sum(len(glob.glob(f"{s}/img1/*")) for s in seqs[:3])
    has_gt = sum(os.path.exists(f"{s}/gt/gt.txt") for s in seqs)
    log(f"{split}: {len(seqs)} sequences | gt.txt present in {has_gt} | first-3 img sample={sample}")

log("DONE")
