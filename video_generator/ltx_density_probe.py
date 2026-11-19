"""Density probe (November 2026 revision): is crowd saturation a capacity limit or a
generalisation / coverage limit?

Renders TRAINING clips the checkpoint was fit on, at the trained 512^2 resolution, and counts
painted instances against ground-truth instances per frame. A token-capacity limit predicts the
same 10-12 painted-instance ceiling on dense clips the model has seen; a generalisation or coverage
limit predicts the model paints the people it was trained on. Resolution is held at 512^2, so
resolution adaptation is not a factor in this probe.

Zero optimizer steps by construction, exactly as video_generator/ltx_valpass.py: the trainer's own
ValidationRunner, sampler parameters parsed from the unmodified phase-A config, no LtxvTrainer,
no optimizer, no load_checkpoint.

Usage: uv run python ltx_density_probe.py <spec.json>
"""
import copy
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import ltx_trainer.video_utils as vu  # noqa: E402
from ltx_trainer.video_utils import _prepare_video_array  # noqa: E402
from ltx_trainer.video_utils import read_video  # noqa: E402

_orig_save_video = vu.save_video


def _save_video_png(video_tensor, output_path, fps=24.0, audio=None, audio_sample_rate=None, video_format=None):
    arr = _prepare_video_array(video_tensor, video_format=video_format)
    d = Path(str(Path(output_path).with_suffix("")) + "_png")
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(arr):
        Image.fromarray(f.astype("uint8")).save(d / f"{i:04d}.png")
    try:
        _orig_save_video(video_tensor, output_path, fps, audio, audio_sample_rate, video_format)
    except Exception as e:  # noqa: BLE001
        print(f"[patch] mp4 save skipped: {e}", flush=True)


vu.save_video = _save_video_png
import ltx_trainer.validation_runner as _vr  # noqa: E402

_vr.save_video = _save_video_png

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from ltx_trainer.config import ValidationConfig  # noqa: E402
from ltx_trainer.model_loader import load_transformer  # noqa: E402
from ltx_trainer.progress import TrainingProgress  # noqa: E402
from ltx_trainer.validation_runner import ValidationRunner  # noqa: E402

spec = json.load(open(sys.argv[1]))
sys.path.insert(0, spec["gentrack_src"])
from gentrack.color_codec import decode as codec_decode  # noqa: E402

K, MIN_AREA = 48, 150


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [density] {m}", flush=True)


OUT, PUB = Path(spec["output_dir"]), Path(spec["publish_root"])
for bad in ("ltx_v2v_ic_lora_phase2", "ltx_v2v_ic_lora_phase3", "ltx_v2v_ic_lora_sm_phase"):
    assert bad not in str(OUT) and bad not in str(PUB), f"refusing to write into {bad}"

raw = yaml.safe_load(open(spec["phase_a_config"]))
vraw = copy.deepcopy(raw["validation"])
tmpl = vraw["samples"][0]
samples = []
for clip in spec["clips"]:                      # pinned list of training-clip RGB references
    e = copy.deepcopy(tmpl)
    for c in e["conditions"]:
        if c.get("type") == "reference":
            c["video"] = clip
    samples.append(e)
vraw["samples"] = samples
vcfg = ValidationConfig.model_validate(vraw)
LORA = raw["lora"]
MODEL_PATH, TE_PATH = raw["model"]["model_path"], raw["model"]["text_encoder_path"]
gts = [c.replace("/rgb/", "/idmap/") for c in spec["clips"]]
missing = [g for g in gts + spec["clips"] if not os.path.exists(g)]
assert not missing, f"MISSING: {missing}"
log(f"{len(samples)} training clips, ckpt {os.path.basename(spec['lora_ckpt'])}")

runner = ValidationRunner(config=vcfg, model_path=MODEL_PATH, text_encoder_path=TE_PATH,
                          load_text_encoder_in_8bit=False)
transformer = load_transformer(MODEL_PATH, device="cuda", dtype=torch.bfloat16)
model = get_peft_model(transformer, LoraConfig(
    r=LORA["rank"], lora_alpha=LORA["alpha"], target_modules=list(LORA["target_modules"]),
    lora_dropout=0.0, init_lora_weights=True))
sd = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(spec["lora_ckpt"]).items()}
set_peft_model_state_dict(model.get_base_model(), sd)
model.eval()
want = float(sum(v.float().abs().sum().item() for k, v in sd.items() if "lora_B" in k))
got = float(sum(p.detach().float().abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n))
assert got > 0 and abs(got - want) <= 1e-3 * max(1.0, want), f"LoRA not resident: {got} vs {want}"
log(f"LoRA applied, |lora_B|={got:.1f}")

with TrainingProgress(enabled=False, total_steps=1) as prog:
    saved = runner.run(model, step=0, output_dir=OUT, device=torch.device("cuda"), progress=prog)
paths = {}
for e in saved:
    if isinstance(e, (tuple, list)) and len(e) == 2:
        paths[int(e[0])] = e[1]
    else:
        paths[len(paths)] = e


def masks_of(rgb):
    return codec_decode(rgb, K=K, min_area=MIN_AREA)


rows = []
for i, clip in enumerate(spec["clips"]):
    png_dir = str(Path(paths[i]).with_suffix("")) + "_png" if i in paths else str(OUT / "samples" / f"step_000000_{i + 1}_png")
    gens = []
    for p in sorted(glob.glob(f"{png_dir}/*.png")):
        f = np.asarray(Image.open(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        gens.append(f)
    gt_t, _ = read_video(gts[i])
    gtf = (gt_t.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    n = min(len(gens), len(gtf))
    painted, gtn, recall = [], [], []
    for t in range(n):
        gm, tm = masks_of(gens[t]), masks_of(gtf[t])
        painted.append(len(gm)); gtn.append(len(tm))
        if tm and gm:
            G, T = list(gm.values()), list(tm.values())
            C = np.zeros((len(T), len(G)))
            for a, tmk in enumerate(T):
                for b, gmk in enumerate(G):
                    inter = np.logical_and(tmk, gmk).sum()
                    C[a, b] = inter / max(np.logical_or(tmk, gmk).sum(), 1)
            r, c = linear_sum_assignment(-C)
            recall.append(float((C[r, c] >= 0.5).sum()) / len(T))
        elif tm:
            recall.append(0.0)
    rows.append(dict(clip=os.path.basename(clip), frames=n,
                     gt_mean=round(float(np.mean(gtn)), 2), gt_max=int(max(gtn)) if gtn else 0,
                     painted_mean=round(float(np.mean(painted)), 2), painted_max=int(max(painted)) if painted else 0,
                     recall_iou50=round(float(np.mean(recall)), 4) if recall else None))
    log(f"{rows[-1]}")

OUT.mkdir(parents=True, exist_ok=True)
res = dict(probe="density on training clips", lora_ckpt=spec["lora_ckpt"], min_area=MIN_AREA, rows=rows,
           optimizer_steps=0, sampler=dict(seed=vcfg.seed, inference_steps=vcfg.inference_steps,
                                           guidance_scale=vcfg.guidance_scale, video_dims=list(vcfg.video_dims)))
json.dump(res, open(OUT / "density_probe.json", "w"), indent=1)
PUB.mkdir(parents=True, exist_ok=True)
shutil.copytree(OUT, PUB, dirs_exist_ok=True)
log(f"DENSITY-DONE -> {PUB}")
