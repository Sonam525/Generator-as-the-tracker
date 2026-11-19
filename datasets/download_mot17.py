# Databricks notebook source
# Download MOT17 train to the volume via DIRECT PARALLEL HTTP (HF resolve URLs). snapshot_
# download of thousands of small LFS files stalls on this cluster; here we fetch each image
# with requests in a thread pool straight to the volume, RESUMABLE (skip existing) with
# incremental progress logged to a volume JSON so we can see the rate. GT comes from the
# parquet mirror (Lekim89/mot17-parquet); images live in the same repo under images/train/.

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

pip("-U", "huggingface_hub", "pandas", "pyarrow", "requests")

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Lekim89/mot17-parquet"
VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/MOT17"
RES = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/results/mot17_find"
os.makedirs(f"{VOL}/train", exist_ok=True)
os.makedirs(RES, exist_ok=True)
out = {}


def progress(done, total, extra=None):
    out["progress"] = {"done": done, "total": total, "t": time.strftime("%H:%M:%S")}
    if extra:
        out.update(extra)
    with open(f"{RES}/download.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


# gt.txt per sequence (vectorized)
gt = pd.read_parquet(hf_hub_download(REPO, "data/gt/train-00000-of-00001.parquet",
                                     repo_type="dataset", token=HF_TOKEN, local_dir="/local_disk0/pq"))
GT_COLS = ["frame", "track_id", "bbox_left", "bbox_top", "bbox_width", "bbox_height",
           "conf", "class_id", "visibility"]
for seq, sdf in gt.groupby("sequence"):
    d = f"{VOL}/train/{seq}/gt"
    os.makedirs(d, exist_ok=True)
    sdf[GT_COLS].to_csv(f"{d}/gt.txt", header=False, index=False)
out["tracks"] = {s: int(g["track_id"].nunique()) for s, g in gt.groupby("sequence")}
log("gt.txt written")

frames = pd.read_parquet(hf_hub_download(REPO, "data/frames/train-00000-of-00001.parquet",
                                         repo_type="dataset", token=HF_TOKEN, local_dir="/local_disk0/pq"))
rows = [(r.sequence, int(r.frame), r.image_path) for r in frames.itertuples()]
out["n_frames"] = len(rows)
log(f"{len(rows)} images to fetch")

sess = requests.Session()
sess.headers["Authorization"] = f"Bearer {HF_TOKEN}"
done = [0]
errs = [0]
lock = threading.Lock()
t0 = time.time()


def fetch(row):
    seq, fr, ip = row
    d = f"{VOL}/train/{seq}/img1"
    os.makedirs(d, exist_ok=True)
    dst = f"{d}/{fr:06d}.jpg"
    if not os.path.exists(dst):
        try:
            url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{ip}"
            r = sess.get(url, timeout=60)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
        except Exception:
            with lock:
                errs[0] += 1
            return
    with lock:
        done[0] += 1
        if done[0] % 300 == 0:
            rate = done[0] / max(1e-6, time.time() - t0)
            log(f"{done[0]}/{len(rows)} ({rate:.1f}/s, {errs[0]} errs)")
            progress(done[0], len(rows), {"errs": errs[0], "rate_per_s": round(rate, 1)})


with ThreadPoolExecutor(max_workers=32) as ex:
    list(ex.map(fetch, rows))

counts = {}
for seq in out["tracks"]:
    p = f"{VOL}/train/{seq}/img1"
    counts[seq] = len(os.listdir(p)) if os.path.isdir(p) else 0
out["frames_written"] = counts
out["errs"] = errs[0]
out["status"] = "OK" if sum(counts.values()) > 0 else "NO_IMAGES"
progress(done[0], len(rows))
log(f"DONE: {counts} ({errs[0]} errs)")
