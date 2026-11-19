# Databricks notebook source
# Generic LTX ic-lora runner: clone + uv sync the trainer, then run our custom entrypoint
# video_generator/ltx_train.py (lossless-PNG validation + early stopping) with a chosen config, and publish
# the (local-disk) output_dir to the volume. Parameterized by the `cfg_ws` widget so the same job
# serves the Phase-1 lossless EVAL and the Phase-2 training-with-early-stopping.

import os
import subprocess
import sys
import time

import yaml

WS = os.environ["GENTRACK_WS"]
VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
ENTRY = f"{WS}/video_generator/ltx_train.py"


def _writable_base():
    """/local_disk0 is not always writable by the notebook user on job clusters (2026-09-07)."""
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
LTX2_SHA = "9377758131b1ffde4b7f766804590a6617bf2ab9"   # pinned: last upstream commit before the July 2026 runs (ships uv.lock)

try:
    dbutils.widgets.text("cfg_ws", f"{WS}/video_generator/configs/ltx_v2v_ic_lora_eval.yaml")  # noqa: F821
    CFG_WS = dbutils.widgets.get("cfg_ws")  # noqa: F821
except Exception:
    CFG_WS = f"{WS}/video_generator/configs/ltx_v2v_ic_lora_eval.yaml"

stem = os.path.splitext(os.path.basename(CFG_WS))[0]
LOGDIR = f"{VOL}/results/{stem}"
os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = f"{LOGDIR}/run.log"
open(LOGFILE, "w").close()
os.environ["PYTHONUNBUFFERED"] = "1"

cfg = yaml.safe_load(open(CFG_WS))
OUT_LOCAL = cfg["output_dir"]                       # local disk (PyAV/PNG write)
if BASE != "/local_disk0" and OUT_LOCAL.startswith("/local_disk0"):
    OUT_LOCAL = BASE + OUT_LOCAL[len("/local_disk0"):]
    cfg["output_dir"] = OUT_LOCAL
    CFG_RUN = f"{BASE}/{stem}.yaml"
    with open(CFG_RUN, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
else:
    CFG_RUN = CFG_WS
OUT_VOL = f"{VOL}/checkpoints/{stem}"               # published copy


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


for p in (CFG_WS, ENTRY, cfg["model"]["model_path"], cfg["model"]["text_encoder_path"]):
    assert os.path.exists(p), f"MISSING: {p}"
log(f"config={CFG_WS} | scratch={BASE} | output_local={OUT_LOCAL} -> {OUT_VOL}")

try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    pass

if not os.path.isdir(REPO):
    sh(f"git clone --filter=blob:none https://github.com/Lightricks/LTX-2.git {REPO}")
    sh(f"git -C {REPO} checkout {LTX2_SHA}")
sh("pip install -q uv", check=False)
sh("uv sync --frozen" if os.path.exists(f"{REPO}/uv.lock") else "uv sync", cwd=REPO)
log("trainer env ready")

# preprocess the manifest -> latents if not already done (manifest sits next to preprocessed_data_root)
PREPROC = cfg["data"]["preprocessed_data_root"]
if not os.path.isdir(f"{PREPROC}/latents"):
    MANIFEST = os.path.join(os.path.dirname(PREPROC), "metadata.jsonl")
    assert os.path.exists(MANIFEST), f"MISSING manifest: {MANIFEST}"
    sh(f'uv run python packages/ltx-trainer/scripts/process_dataset.py "{MANIFEST}" '
       f'--resolution-buckets "512x512x49" --model-path "{cfg["model"]["model_path"]}" '
       f'--text-encoder-path "{cfg["model"]["text_encoder_path"]}" --skip-audio '
       f'--output-dir "{PREPROC}"', cwd=REPO)
    log("preprocess done")
else:
    log("latents present — skipping preprocess")

try:
    sh(f'uv run python "{ENTRY}" "{CFG_RUN}"', cwd=REPO)
    log("ENTRYPOINT-DONE")
finally:
    sh(f'mkdir -p "{OUT_VOL}" && cp -r "{OUT_LOCAL}/." "{OUT_VOL}/" 2>/dev/null || true', check=False)
    log(f"published -> {OUT_VOL}")
