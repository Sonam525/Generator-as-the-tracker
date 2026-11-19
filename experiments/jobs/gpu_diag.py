# Databricks notebook source
# Job-cluster diagnostic (2026-09-07): GPU visibility + scratch-disk permissions. Cheap; no data touched.
# Findings are returned via dbutils.notebook.exit (readable with `jobs get-run-output`) and also
# written to results/gpu_diag/<host>.txt on the volume.
import os
import socket
import subprocess

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
BUF = []
try:
    os.makedirs(f"{VOL}/results/gpu_diag", exist_ok=True)
    LOG = open(f"{VOL}/results/gpu_diag/{socket.gethostname()}.txt", "w")
except Exception:  # noqa: BLE001
    LOG = None


def out(m):
    print(m, flush=True)
    BUF.append(m)
    if LOG:
        LOG.write(m + "\n"); LOG.flush()


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    out(f"$ {c}\n[rc={r.returncode}]\n{(r.stdout or '')[-1500:]}{(r.stderr or '')[-600:]}")


sh("whoami; id; hostname; uname -r; head -2 /etc/os-release")
sh("nvidia-smi -L; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
sh("ls -l /dev/nvidia* 2>&1 | head; ls -l /usr/lib/x86_64-linux-gnu/libcuda.so* 2>&1 | head -3")
sh("python -c \"import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')\"")
sh("ls -ld /local_disk0 /local_disk0/* 2>&1 | head -8; df -h /local_disk0 /tmp / 2>&1 | head -6")
for base in ("/local_disk0", "/local_disk0/ltxwork", "/tmp/ltxwork"):
    try:
        os.makedirs(base, exist_ok=True)
        p = os.path.join(base, ".probe")
        open(p, "w").close(); os.remove(p)
        out(f"WRITABLE {base}")
    except Exception as e:  # noqa: BLE001
        out(f"NOT WRITABLE {base}: {e}")
sh("sudo -n true && echo SUDO_OK || echo SUDO_NO")
sh("env | grep -i -E 'cuda|nvidia|spark.master|DB_IS|DATABRICKS_RUNTIME' | head -20")
out("DIAG-DONE")
if LOG:
    LOG.close()
try:
    dbutils.notebook.exit("\n".join(BUF)[:20000])  # noqa: F821
except Exception:  # noqa: BLE001
    pass
