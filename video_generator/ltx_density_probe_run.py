# Databricks notebook source
# Density probe runner (November 2026 revision): clone the pinned LTX-2 trainer, then run
# video_generator/ltx_density_probe.py, which drives the trainer's ValidationRunner directly on training clips. NO training: no LtxvTrainer, no optimizer,
# no latents, no load_checkpoint. See ltx_valpass.py for why the config route is unsafe here.
# Widgets are read one default each (a shared try/except silently ran the wrong experiment on
# 2026-09-08).

import json
import os
import subprocess
import time

WS = os.environ["GENTRACK_WS"]
VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
ENTRY = f"{WS}/video_generator/ltx_density_probe.py"
LTX2_SHA = "9377758131b1ffde4b7f766804590a6617bf2ab9"


def _writable_base():
    import getpass
    for base in ("/local_disk0", "/local_disk0/ltxwork", "/tmp/ltxwork",
                 f"/tmp/ltxwork_{os.getpid()}", os.path.expanduser("~/ltxwork")):
        try:
            os.makedirs(base, exist_ok=True)
            probe = os.path.join(base, ".w")
            open(probe, "w").close()
            os.remove(probe)
            return base
        except Exception:  # noqa: BLE001
            if base == "/local_disk0/ltxwork":
                r = subprocess.run(
                    f"sudo -n mkdir -p {base} && sudo -n chown -R {getpass.getuser()} {base}",
                    shell=True, capture_output=True, text=True)
                if r.returncode == 0 and os.access(base, os.W_OK):
                    return base
    raise RuntimeError("no writable local scratch directory")


BASE = _writable_base()
REPO = f"{BASE}/LTX-2"


def _widget(name, default):
    """One widget, one default: a missing key must not revert the others."""
    try:
        v = dbutils.widgets.get(name)  # noqa: F821
    except Exception:  # noqa: BLE001
        return default
    return v if v not in (None, "") else default


PHASE_A_CFG = _widget("phase_a_cfg", f"{WS}/video_generator/configs/ltx_v2v_ic_lora_phase2.yaml")
LORA_CKPT = _widget("lora_ckpt", f"{VOL}/checkpoints/ltx_v2v_ic_lora_phase2/checkpoints/"
                                 f"lora_weights_step_01500.safetensors")
CLIPS = [c for c in _widget("clips", "").split("|") if c]     # pinned training clips, "|"-separated
OUT_NAME = _widget("out_name", "ltx_dt_density_probe")
assert CLIPS, "no clips given"

OUT_LOCAL = f"{BASE}/ltx_out/{OUT_NAME}"
PUB = f"{VOL}/checkpoints/{OUT_NAME}"
LOGDIR = f"{VOL}/results/{OUT_NAME}"
os.makedirs(LOGDIR, exist_ok=True)
LOGFILE = f"{LOGDIR}/run.log"
open(LOGFILE, "w").close()
os.environ["PYTHONUNBUFFERED"] = "1"

_BUF = []


def _flush():
    """Rewrite the whole file: appending to a UC volume does not materialise past the first write."""
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


for p in [ENTRY, PHASE_A_CFG, LORA_CKPT] + CLIPS:
    assert os.path.exists(p), f"MISSING: {p}"
log(f"density probe | {len(CLIPS)} clips | ckpt={os.path.basename(LORA_CKPT)} | scratch={BASE} -> {PUB}")

try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:  # noqa: BLE001
    pass

if not os.path.isdir(REPO):
    sh(f"git clone --filter=blob:none https://github.com/Lightricks/LTX-2.git {REPO}")
    sh(f"git -C {REPO} checkout {LTX2_SHA}")
sh("pip install -q uv", check=False)
sh("uv sync --frozen" if os.path.exists(f"{REPO}/uv.lock") else "uv sync", cwd=REPO)
sh("uv pip install -q scikit-image scipy", cwd=REPO)
log("trainer env ready")

SPEC = dict(phase_a_config=PHASE_A_CFG, lora_ckpt=LORA_CKPT, clips=CLIPS, output_dir=OUT_LOCAL,
            publish_root=PUB, gentrack_src=f"{WS}/core")
SPEC_PATH = f"{BASE}/density_spec.json"
with open(SPEC_PATH, "w") as f:
    json.dump(SPEC, f, indent=2)

try:
    sh(f'uv run python "{ENTRY}" "{SPEC_PATH}"', cwd=REPO)
finally:
    sh(f'mkdir -p "{PUB}" && cp -r "{OUT_LOCAL}/." "{PUB}/" 2>/dev/null || true', check=False)
log("DENSITY-RUN-DONE")
