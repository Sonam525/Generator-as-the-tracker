# Databricks notebook source
# Build the DanceTrack TEST-SERVER submission from the test chain outputs (CPU, no GT anywhere).
# Per test video: decode cleaned windows -> raw-colour global ids (Arm-A frozen method) -> boxes
# scaled 512 -> original resolution -> MOTChallenge txt (frame,id,x,y,w,h,1,-1,-1,-1).
# Produces submission zip (txts at zip root) + a sanity manifest (coverage, id/box counts).

import glob
import json
import os
import subprocess
import sys
import time
import zipfile
from multiprocessing import Pool


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy"], check=True)
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
ARM = "ltx_test_chain"
RES = f"{VOL}/results/ltx_test_submission"
os.makedirs(RES, exist_ok=True)
K, MIN_AREA = 48, 150
STAGE = "/local_disk0/submission"
os.makedirs(STAGE, exist_ok=True)

vids = sorted(os.path.basename(p) for p in glob.glob(f"{VOL}/checkpoints/{ARM}/dancetrack*"))
log(f"test videos with outputs: {len(vids)}")


def process_video(vid):
    palette = build(K)
    meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{vid}_ov9/windows.json"))
    wins = sorted(meta["windows"], key=lambda w: w["idx"])
    total = meta["total_frames"]
    w0, h0 = meta["image_size"]
    own_start = [wins[0]["start"]]
    for k in range(1, len(wins)):
        own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
    own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]

    acc = {}                                            # (frame, colour_id) -> union box (512 space)
    for wi, w in enumerate(wins):
        frames = [np.asarray(imageio.imread(p))[:, :, :3]
                  for p in sorted(glob.glob(f"{VOL}/checkpoints/{ARM}/{vid}/cleaned/w{wi:02d}/*.png"))]
        if not frames:
            return dict(vid=vid, error=f"no cleaned frames w{wi:02d}")
        for b in idmaps_to_boxes(frames, palette=palette, min_area=MIN_AREA):
            gf = w["start"] + (b.frame - 1)
            if not (own_start[wi] <= gf <= own_end[wi]):
                continue
            key = (gf, b.track_id)                      # raw colour = global identity (Arm A)
            if key in acc:
                x0 = min(acc[key][0], b.x); y0 = min(acc[key][1], b.y)
                x1 = max(acc[key][0] + acc[key][2], b.x + b.w)
                y1 = max(acc[key][1] + acc[key][3], b.y + b.h)
                acc[key] = (x0, y0, x1 - x0, y1 - y0)
            else:
                acc[key] = (b.x, b.y, b.w, b.h)

    ux, uy = w0 / 512.0, h0 / 512.0
    lines = []
    frames_covered = set()
    ids = set()
    for (gf, gid), (x, y, w_, h_) in sorted(acc.items()):
        lines.append(f"{gf},{gid},{x*ux:.2f},{y*uy:.2f},{w_*ux:.2f},{h_*uy:.2f},1,-1,-1,-1")
        frames_covered.add(gf); ids.add(gid)
    with open(f"{STAGE}/{vid}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    return dict(vid=vid, frames=total, frames_covered=len(frames_covered),
                coverage=round(len(frames_covered) / total, 3), n_ids=len(ids), n_boxes=len(lines))


with Pool(4) as pool:
    results = pool.map(process_video, vids)

manifest, bad = [], []
for r in results:
    if "error" in r:
        bad.append(r); log(f"{r['vid']}: ERROR {r['error']}"); continue
    manifest.append(r)
    log(f"{r['vid']}: {r['n_boxes']} boxes, {r['n_ids']} ids, frame coverage {r['coverage']} "
        f"({r['frames_covered']}/{r['frames']})")
assert not bad, f"failed videos: {[b['vid'] for b in bad]}"
assert len(manifest) == 35, f"expected 35 sequences, got {len(manifest)}"

zpath = f"{STAGE}/gentrack_ltx_test_submission.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for r in manifest:
        z.write(f"{STAGE}/{r['vid']}.txt", arcname=f"{r['vid']}.txt")
import shutil  # noqa: E402
shutil.copy(zpath, f"{RES}/gentrack_ltx_test_submission.zip")
for r in manifest:
    shutil.copy(f"{STAGE}/{r['vid']}.txt", f"{RES}/{r['vid']}.txt")
with open(f"{RES}/manifest.json", "w") as f:
    json.dump(dict(arm=ARM, method="chained phase-3, raw-colour identity (frozen from val)",
                   n_sequences=len(manifest),
                   mean_coverage=round(float(np.mean([r['coverage'] for r in manifest])), 3),
                   per_video=manifest), f, indent=2)
log(f"SUBMISSION READY -> {RES}/gentrack_ltx_test_submission.zip "
    f"(mean frame coverage {np.mean([r['coverage'] for r in manifest]):.3f})")
