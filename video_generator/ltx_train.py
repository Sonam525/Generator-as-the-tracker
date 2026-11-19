"""Custom LTX ic-lora train/eval entrypoint (run inside the trainer's uv venv via `uv run`).

Two additions over the stock scripts/train.py:
  1. LOSSLESS validation frames — monkeypatch save_video so each validation generation is also
     dumped as per-frame PNG (the stock path writes yuv420p mp4, which smears flat ID-map colours
     and breaks codec decode — working-agreement rule 2, README.md). PNGs go next to the mp4 in `<name>_png/`.
  2. EARLY STOPPING — a step_callback computes a foreground-IoU proxy (non-black overlap) between
     each validation generation and its GT id-map, tracks the best, writes BEST_STEP.txt, and stops
     (patience) when it plateaus. GT id-map is derived from the sample's reference path (rgb/->idmap/).
     Proxy metric (numpy only, no gentrack deps in the venv); the precise codec IoU gate is a
     separate decode step on the PNGs.

Usage: uv run python ltx_train.py <config.yaml>    (ES_PATIENCE env overrides patience, default 3)
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# --- (1) lossless-PNG patch: must run before the trainer imports validation ------------------
import ltx_trainer.video_utils as vu  # noqa: E402
from ltx_trainer.video_utils import _prepare_video_array  # noqa: E402
from ltx_trainer.video_utils import read_video  # noqa: E402

_orig_save_video = vu.save_video


def _save_video_png(video_tensor, output_path, fps=24.0, audio=None,
                    audio_sample_rate=None, video_format=None):
    arr = _prepare_video_array(video_tensor, video_format=video_format)  # [F,H,W,C] uint8
    d = Path(str(Path(output_path).with_suffix("")) + "_png")
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(arr):
        Image.fromarray(f.astype("uint8")).save(d / f"{i:04d}.png")
    try:  # keep the mp4 too (harmless on local disk); PNGs are the lossless artifact we use
        _orig_save_video(video_tensor, output_path, fps, audio, audio_sample_rate, video_format)
    except Exception as e:
        print(f"[patch] mp4 save skipped: {e}", flush=True)


vu.save_video = _save_video_png
import ltx_trainer.validation_runner as _vr  # noqa: E402
_vr.save_video = _save_video_png  # validation_runner imported the name directly

from ltx_trainer.config import LtxTrainerConfig  # noqa: E402
from ltx_trainer.trainer import LtxvTrainer  # noqa: E402

cfg_path = sys.argv[1]
cfg = LtxTrainerConfig(**yaml.safe_load(open(cfg_path)))
trainer = LtxvTrainer(cfg)

# --- (2) early stopping on validation foreground-IoU -----------------------------------------
PATIENCE = int(os.environ.get("ES_PATIENCE", "3"))
MIN_DELTA = 0.003
# the trainer calls step_callback EVERY optimization step (re-using stale validation paths between
# validations) -> only act on actual validation steps, so `since` counts validations, not steps.
VAL_INTERVAL = int(getattr(cfg.validation, "interval", 0) or 0)
samples = list(getattr(cfg.validation, "samples", []) or [])
best = {"m": -1.0, "step": -1, "since": 0}


class StopTraining(Exception):
    pass


ES_LOG = f"{cfg.output_dir}/earlystop.jsonl"   # local disk -> published to the volume by ltx_run


def _es_write(rec):
    try:
        with open(ES_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _gt_idmap_for(sample):
    for c in (getattr(sample, "conditions", None) or []):
        v = getattr(c, "video", None) or (c.get("video") if isinstance(c, dict) else None)
        if v:
            return str(v).replace("/rgb/", "/idmap/")
    return None


def _fg_iou(gen_png_dir, gt_path):
    """Colour-blind foreground IoU of the RIGHT half (the id-map) vs GT. Returns (iou, why)."""
    gps = sorted(glob.glob(f"{gen_png_dir}/*.png"))
    if not gps:
        return None, "no_png"
    if not gt_path or not os.path.exists(gt_path):
        return None, f"no_gt"
    gen = []
    for p in gps:
        f = np.asarray(Image.open(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:          # composite [ref|gen] -> right half
            f = f[:, f.shape[1] // 2:, :]
        gen.append(f)
    gt_t, _ = read_video(gt_path)
    gt = (gt_t.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    n = min(len(gen), len(gt))
    ious = []
    for i in range(n):
        a = gen[i].sum(2) > 30
        b = gt[i].sum(2) > 30
        u = (a | b).sum()
        if u > 0:
            ious.append(float((a & b).sum()) / float(u))
    return (float(np.mean(ious)) if ious else 0.0), "ok"


def early_stop_cb(step, total, sampled_paths):
    if not sampled_paths or (VAL_INTERVAL and step % VAL_INTERVAL != 0):
        return
    per, ms = {}, []
    for i, p in enumerate(sampled_paths):
        png_dir = str(Path(p).with_suffix("")) + "_png"
        gt = _gt_idmap_for(samples[i]) if i < len(samples) else None
        try:
            m, why = _fg_iou(png_dir, gt)
        except Exception as e:
            m, why = None, f"err:{type(e).__name__}"
        per[os.path.basename(png_dir)] = {"iou": (round(m, 3) if m is not None else None), "why": why}
        if m is not None:
            ms.append(m)
    val = round(float(np.mean(ms)), 4) if ms else None
    if val is not None and val > best["m"] + MIN_DELTA:
        best.update(m=val, step=step, since=0)
        try:
            open(f"{cfg.output_dir}/BEST_STEP.txt", "w").write(f"step={step} val_fg_iou={val:.4f}\n")
        except Exception:
            pass
    elif val is not None:
        best["since"] += 1
    _es_write({"step": int(step), "mean_fg_iou": val, "n_ok": len(ms), "n_samples": len(sampled_paths),
               "best": round(best["m"], 4), "best_step": best["step"], "since": best["since"], "per": per})
    print(f"[earlystop] step {step} val_fg_iou={val} best={best['m']:.3f}@{best['step']} "
          f"since={best['since']}/{PATIENCE} (scored {len(ms)}/{len(sampled_paths)})", flush=True)
    if val is not None and best["since"] >= PATIENCE:
        raise StopTraining()


try:
    trainer.train(step_callback=early_stop_cb)
except StopTraining:
    print("[earlystop] training stopped early.", flush=True)
print(f"LTX-TRAIN-DONE best_step={best['step']} best_val_fg_iou={best['m']:.3f}", flush=True)
