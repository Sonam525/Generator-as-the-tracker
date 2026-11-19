# Databricks notebook source
# Runner for multi-video windowed generation (ltx_chain.py), both arms:
#   mode=chain        -> phase-3 ckpt, prefix conditioning (generator-carried identity)
#   mode=independent  -> phase-2 ckpt, reference-only windows (post-hoc v3 stitcher at eval)
# Widgets: vids ("all25" or comma list), mode, lora_ckpt, out_name, win_suffix.
# Incremental per-video publish + resume happens inside ltx_chain.py (DONE.txt markers).

import json
import os
import subprocess
import time

WS = os.environ["GENTRACK_WS"]
VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
ENTRY = f"{WS}/video_generator/ltx_chain.py"


def _writable_base():
    """Local scratch for the trainer checkout and window outputs. /local_disk0 is not always
    writable by the notebook user on job clusters (seen 2026-09-07 on NC24ads_A100_v4):
    try it, then a sudo-chowned subdir, then /tmp (driver disk; outputs are published to the
    volume per video anyway)."""
    import getpass
    for base in ("/local_disk0", "/local_disk0/ltxwork", "/tmp/ltxwork", f"/tmp/ltxwork_{os.getpid()}",
                 os.path.expanduser("~/ltxwork")):
        try:
            os.makedirs(base, exist_ok=True)
            probe = os.path.join(base, ".w")
            open(probe, "w").close(); os.remove(probe)
            return base
        except Exception:
            if base == "/local_disk0/ltxwork":
                r = subprocess.run(f"sudo -n mkdir -p {base} && sudo -n chown -R {getpass.getuser()} {base}",
                                   shell=True, capture_output=True, text=True)
                if r.returncode == 0 and os.access(base, os.W_OK):
                    return base
    raise RuntimeError("no writable local scratch directory")


BASE = _writable_base()
REPO = f"{BASE}/LTX-2"

ALL25 = ["dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
         "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
         "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
         "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
         "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097"]
P3_CKPT = f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase3/checkpoints/lora_weights_step_02500.safetensors"
P2_CKPT = f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase2/checkpoints/lora_weights_step_01500.safetensors"

def _widget(name, default):
    """One widget, one default: a widget missing from base_parameters must not revert the others.
    `size`/`seed` are not declared below, so under a shared try/except a job that omitted either
    would fall all the way back to all25 DanceTrack -- hours of A100 time on the wrong videos."""
    try:
        v = dbutils.widgets.get(name)  # noqa: F821
    except Exception:  # noqa: BLE001
        return default
    return v if v not in (None, "") else default


VIDS_IN = _widget("vids", "all25")
MODE = _widget("mode", "chain")
LORA_CKPT = _widget("lora_ckpt", "")
OUT_NAME = _widget("out_name", "ltx_arm_a_chain")
SUFFIX = _widget("win_suffix", "_ov9")
SIZE = int(_widget("size", "512"))
SEED = int(_widget("seed", "0"))            # sampling seed (seed-variance runs)

VIDEOS = ALL25 if VIDS_IN.strip().lower() == "all25" else [v.strip() for v in VIDS_IN.split(",") if v.strip()]
if not LORA_CKPT:
    LORA_CKPT = P3_CKPT if MODE == "chain" else P2_CKPT

OUT_LOCAL = f"{BASE}/ltx_out/{OUT_NAME}"
PUB = f"{VOL}/checkpoints/{OUT_NAME}"
LOGDIR = f"{VOL}/results/{OUT_NAME}"
os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = f"{LOGDIR}/run.log"
open(LOGFILE, "w").close()
os.environ["PYTHONUNBUFFERED"] = "1"

SPEC = dict(
    mode=MODE,
    model_path=f"{VOL}/models/ltx2_native/model/ltx-2.3-22b-dev.safetensors",
    text_encoder_path=f"{VOL}/models/ltx2_native/gemma",
    lora_checkpoint=LORA_CKPT,
    lora_rank=32, lora_alpha=32,
    lora_targets=["attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                  "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
                  "ff.net.0.proj", "ff.net.2"],
    videos=VIDEOS,
    windows_root=f"{VOL}/ltx_ds/fullseq",
    win_suffix=SUFFIX,
    output_dir=OUT_LOCAL,
    publish_root=PUB,
    gentrack_src=f"{WS}/core",
    prompt="segmentation ID map: paint each person a distinct flat colour on a black background",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    video_dims=[SIZE, SIZE, 49], seed=SEED, inference_steps=20,
    guidance_scale=4.0, stg_scale=1.0, stg_blocks=[29], stg_mode="stg_v",
    prefix_frames=9, K=48, min_area=150,
)


_BUF = []


def _flush():
    """Rewrite the whole file. Appending to a UC volume does not materialise past the first
    write (verified 2026-09-08: three successful runs each left a 9-line run.log stopping at
    the git clone), which makes every long run look hung. The log is small; rewriting is cheap."""
    try:
        with open(LOGFILE, "w") as f:
            f.write("\n".join(_BUF) + "\n")
    except Exception:  # noqa: BLE001
        pass


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    _BUF.append(line)
    _flush()


def sh(cmd, cwd=None, check=True):
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out, err = (r.stdout or "")[-20000:], (r.stderr or "")[-20000:]
    _BUF.append(f"===== $ {cmd}\n[rc={r.returncode}]\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    _flush()
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n"
                           f"--- STDOUT tail ---\n{out[-4000:]}\n--- STDERR tail ---\n{err[-4000:]}")
    return r


for p in (ENTRY, SPEC["model_path"], SPEC["text_encoder_path"], SPEC["lora_checkpoint"]):
    assert os.path.exists(p), f"MISSING: {p}"
missing = [v for v in VIDEOS if not os.path.exists(f"{VOL}/ltx_ds/fullseq/{v}{SUFFIX}/windows.json")]
assert not missing, f"MISSING windows.json for: {missing}"
log(f"mode={MODE} ckpt={os.path.basename(LORA_CKPT)} seed={SEED} videos={len(VIDEOS)} scratch={BASE} -> {PUB}")

LTX2_SHA = "9377758131b1ffde4b7f766804590a6617bf2ab9"   # pinned: last upstream commit before the July 2026 runs (ships uv.lock)
if not os.path.isdir(REPO):
    sh(f"git clone --filter=blob:none https://github.com/Lightricks/LTX-2.git {REPO}")
    sh(f"git -C {REPO} checkout {LTX2_SHA}")
sh("pip install -q uv", check=False)
sh("uv sync --frozen" if os.path.exists(f"{REPO}/uv.lock") else "uv sync", cwd=REPO)
sh("uv pip install -q scikit-image scipy imageio imageio-ffmpeg", cwd=REPO)
log("trainer env ready")

SPEC_PATH = f"{BASE}/chain_spec.json"
with open(SPEC_PATH, "w") as f:
    json.dump(SPEC, f, indent=2)

sh(f'uv run python "{ENTRY}" "{SPEC_PATH}"', cwd=REPO)
log("ARM-DONE")
