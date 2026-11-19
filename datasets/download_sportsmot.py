# Databricks notebook source
# Download SportsMOT (Cui et al., ICCV 2023) from Hugging Face to the UC volume, MOT-17 layout.
# Source: https://huggingface.co/datasets/MCG-NJU/SportsMOT  (cc-by-nc-4.0; dataset/{train,val,test}.tar
# + splits_txt/*.txt). Default = train + val only; test.tar (23 GB, 150 videos) is a separate,
# spend-gated step (working-agreement rule 7, README.md: submission server at the end only).
# Detached CPU job; resumable (HF cache resume + per-tar done markers). Extraction to the blob
# volume is write-latency-bound, so tars are extracted on the driver disk and moved per sequence.

import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    from huggingface_hub import hf_hub_download
except ImportError:
    _pip("huggingface_hub")
    from huggingface_hub import hf_hub_download

REPO = "MCG-NJU/SportsMOT"
ROOT = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker") + "/sportsmot"
try:
    SPLITS = [s.strip() for s in dbutils.widgets.get("splits").split(",") if s.strip()]  # noqa: F821
except Exception:
    SPLITS = ["train", "val"]
TOKEN = os.environ.get("HF_TOKEN")


def _stage():
    for base in ("/local_disk0", "/tmp"):
        try:
            d = f"{base}/sportsmot_stage"
            os.makedirs(d, exist_ok=True)
            open(f"{d}/.w", "w").close(); os.remove(f"{d}/.w")
            return d
        except Exception:
            continue
    raise RuntimeError("no writable scratch")


STAGE = _stage()
os.makedirs(ROOT, exist_ok=True)
log(f"stage={STAGE} root={ROOT} splits={SPLITS}")

# split lists (video -> split, sport)
os.makedirs(f"{ROOT}/splits_txt", exist_ok=True)
for name in ("basketball.txt", "football.txt", "volleyball.txt", "train.txt", "val.txt", "test.txt"):
    try:
        p = hf_hub_download(repo_id=REPO, filename=f"splits_txt/{name}", repo_type="dataset",
                            local_dir=STAGE, token=TOKEN)
        shutil.copy(p, f"{ROOT}/splits_txt/{name}")
    except Exception as e:  # noqa: BLE001
        log(f"splits_txt/{name}: not fetched ({str(e)[:120]})")

for split in SPLITS:
    marker = pathlib.Path(ROOT) / f".done_{split}"
    if marker.exists():
        log(f"{split}: already extracted (marker present), skipping")
        continue
    fname = f"dataset/{split}.tar"
    log(f"{split}: downloading {fname} from {REPO} ...")
    tp = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset", local_dir=STAGE, token=TOKEN)
    log(f"{split}: {os.path.getsize(tp) / 1e9:.2f} GB -> {tp}; extracting on the driver disk ...")
    ext = f"{STAGE}/{split}_extract"
    os.makedirs(ext, exist_ok=True)
    with tarfile.open(tp) as t:
        names = t.getnames()
        log(f"{split}: {len(names)} members; first entries {names[:3]}")
        t.extractall(ext)
    # find sequence dirs (those containing img1/) wherever the tar nested them, move to ROOT/split/
    seq_dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{ext}/**/img1", recursive=True)})
    log(f"{split}: {len(seq_dirs)} sequence dirs found; moving to {ROOT}/{split}/")
    os.makedirs(f"{ROOT}/{split}", exist_ok=True)
    t0 = time.time()
    for i, sd in enumerate(seq_dirs, 1):
        dest = f"{ROOT}/{split}/{os.path.basename(sd)}"
        if os.path.isdir(dest) and os.path.exists(f"{dest}/seqinfo.ini"):
            continue
        shutil.copytree(sd, dest, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("._*", ".DS_Store"))   # AppleDouble sidecars in the tars
        if i % 5 == 0:
            log(f"  {i}/{len(seq_dirs)} moved ({time.time() - t0:.0f}s)")
    shutil.rmtree(ext, ignore_errors=True)
    try:
        os.remove(tp)
    except OSError:
        pass
    marker.write_text("ok")
    log(f"{split}: done, marker written")

# Verify
for split in SPLITS:
    seqs = sorted(glob.glob(f"{ROOT}/{split}/*/"))
    seqs = [s for s in seqs if os.path.isdir(f"{s}/img1")]
    frames = [len(glob.glob(f"{s}/img1/*")) for s in seqs]
    has_gt = sum(os.path.exists(f"{s}/gt/gt.txt") for s in seqs)
    log(f"{split}: {len(seqs)} sequences | gt.txt in {has_gt} | frames total {sum(frames)} "
        f"min {min(frames) if frames else 0} max {max(frames) if frames else 0}")
    if seqs:
        log(f"{split}: seqinfo sample:\n" + open(f"{seqs[0]}/seqinfo.ini").read()[:300])
