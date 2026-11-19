"""G-SM2 literal scoring: one validation-style render of the 8 SportsMOT val windows per LoRA arm.

Gate G-SM2 (pre-registered 2026-09-07) reads "phase-A val fg-IoU rises above the zero-shot ckpt".
Phase A recorded mean val fg-IoU 0.5547 at step 750 over 8 fixed SportsMOT val windows, produced by
the trainer's own ValidationRunner and scored by _fg_iou in video_generator/ltx_train.py. The zero-shot
checkpoint was only ever run as a chain, which keeps cleaned maps rather than raw renders, so the
literal comparison needs one validation pass with that checkpoint on the same 8 samples.

Zero optimizer steps BY CONSTRUCTION, not by configuration: this file never builds an LtxvTrainer,
an optimizer or a training dataloader, never reads data.preprocessed_data_root, and never touches
model.load_checkpoint. Going through ltx_run.py with a config instead would have been unsafe: a
training_state_step_*.pt sits beside every lora_weights file and load_checkpoint restores
global_step, so that route resumes and trains, silently voiding the zero-shot status.

Comparability is enforced rather than asserted:
  * every sampler parameter is inherited by parsing the UNMODIFIED phase-A config, never retyped;
  * the metric is lifted verbatim out of video_generator/ltx_train.py by source slice and exec'd, with the
    slice's sha256 recorded, so no threshold can drift;
  * arm 1 is a CONTROL, the phase-A step-750 LoRA itself. The renders behind 0.5547 were reproduced
    byte-identically by a later job on a different cluster (sm_phase2 and sm_phase3
    samples/step_000750_1_png/0000.png share sha256 e9e6c8de3763b0fb...), so this sampler is
    bit-deterministic and the control must return exactly 0.5547. If it does not, the apparatus is
    not faithful and the gate is UNSCORED rather than reported.

Usage: uv run python ltx_valpass.py <valpass_spec.json>
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# lossless-PNG capture of every render, exactly as video_generator/ltx_train.py does it; must be patched
# before validation_runner is used, and on both module objects because it imports the name directly.
import ltx_trainer.video_utils as vu  # noqa: E402
from ltx_trainer.video_utils import _prepare_video_array  # noqa: E402
from ltx_trainer.video_utils import read_video  # noqa: E402

_orig_save_video = vu.save_video


def _save_video_png(video_tensor, output_path, fps=24.0, audio=None,
                    audio_sample_rate=None, video_format=None):
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

from ltx_trainer.model_loader import load_transformer  # noqa: E402
from ltx_trainer.progress import TrainingProgress  # noqa: E402
from ltx_trainer.validation_runner import ValidationRunner  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [valpass] {m}", flush=True)


spec = json.load(open(sys.argv[1]))
OUT, PUB = Path(spec["output_dir"]), Path(spec["publish_root"])
for bad in ("ltx_v2v_ic_lora_sm_phase2", "ltx_v2v_ic_lora_sm_phase3", "ltx_v2v_ic_lora_phase3"):
    assert bad not in str(OUT) and bad not in str(PUB), f"refusing to write into {bad}"

raw = yaml.safe_load(open(spec["phase_a_config"]))
try:
    from ltx_trainer.config import ValidationConfig
    vcfg = ValidationConfig.model_validate(raw["validation"])
except Exception as e:  # noqa: BLE001
    log(f"ValidationConfig.model_validate failed ({e}); falling back to the full trainer config")
    from ltx_trainer.config import LtxTrainerConfig
    vcfg = LtxTrainerConfig(**raw).validation
samples = list(vcfg.samples)
assert len(samples) == 8, f"expected the 8 pinned val samples, got {len(samples)}"
LORA = raw["lora"]
MODEL_PATH, TE_PATH = raw["model"]["model_path"], raw["model"]["text_encoder_path"]

# metric lifted verbatim from the training entrypoint, never re-implemented here
tsrc = open(spec["train_entry"], encoding="utf-8").read()
i, j = tsrc.index("def _gt_idmap_for("), tsrc.index("def early_stop_cb(")
metric_src = tsrc[i:j]
METRIC_SHA = hashlib.sha256(metric_src.encode()).hexdigest()[:16]
_ns = {"glob": glob, "os": os, "np": np, "Image": Image, "read_video": read_video}
exec(compile(metric_src, spec["train_entry"], "exec"), _ns)
_gt_idmap_for, _fg_iou = _ns["_gt_idmap_for"], _ns["_fg_iou"]
log(f"metric lifted from {spec['train_entry']} sha256[:16]={METRIC_SHA}")

gts = [_gt_idmap_for(s) for s in samples]
missing = [g for g in gts if not g or not os.path.exists(g)]
assert not missing, f"MISSING GT id-maps: {missing}"
gt_stat = {os.path.basename(g): {"bytes": os.path.getsize(g), "mtime": round(os.path.getmtime(g))}
           for g in gts}

REFERENCE = 0.5547
results = {
    "gate": "G-SM2", "metric_sha256_16": METRIC_SHA, "phase_a_config": spec["phase_a_config"],
    "reference": {"run": "SportsMOT phase A step 750", "mean_fg_iou": REFERENCE},
    "sampler": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in
                ((n, getattr(vcfg, n, None)) for n in
                 ("video_dims", "seed", "inference_steps", "guidance_scale", "stg_scale",
                  "stg_blocks", "stg_mode", "frame_rate", "negative_prompt"))},
    "gt_files": gt_stat, "optimizer_steps": 0, "arms": {}, "arm_order": [a["name"] for a in spec["arms"]],
}


def write_results():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "g_sm2_valpass.json", "w") as fh:
        json.dump(results, fh, indent=2)


t0 = time.time()
runner = ValidationRunner(config=vcfg, model_path=MODEL_PATH, text_encoder_path=TE_PATH,
                          load_text_encoder_in_8bit=False)
log(f"runner ready in {time.time() - t0:.0f}s (conditioning media cached for 8 samples)")

t0 = time.time()
transformer = load_transformer(MODEL_PATH, device="cuda", dtype=torch.bfloat16)
model = get_peft_model(transformer, LoraConfig(
    r=LORA["rank"], lora_alpha=LORA["alpha"], target_modules=list(LORA["target_modules"]),
    lora_dropout=0.0, init_lora_weights=True))
log(f"transformer ready in {time.time() - t0:.0f}s")


def lora_b_fingerprint():
    """Sum of |lora_B| over the live model. Zero before any load; distinct per checkpoint."""
    return float(sum(p.detach().float().abs().sum().item()
                     for n, p in model.named_parameters() if "lora_B" in n))


device = torch.device("cuda")
seen_fp = {}
with TrainingProgress(enabled=False, total_steps=1) as prog:
    for arm in spec["arms"]:
        name, ckpt = arm["name"], arm["lora_ckpt"]
        sd = load_file(ckpt)
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
        set_peft_model_state_dict(model.get_base_model(), sd)
        model.eval()
        # The adapter must actually be resident. A non-zero check alone is vacuous after arm 1
        # (arm 1's weights satisfy it), so tie the live model to THIS file's tensors and require
        # the fingerprint to differ from every arm already run.
        want = float(sum(v.float().abs().sum().item() for k, v in sd.items() if "lora_B" in k))
        got = lora_b_fingerprint()
        assert got > 0, f"{name}: lora_B all zero after load, the LoRA did not apply"
        assert abs(got - want) <= 1e-3 * max(1.0, want), (
            f"{name}: live |lora_B|={got:.3f} does not match the checkpoint's {want:.3f}")
        for prev, fp in seen_fp.items():
            assert abs(got - fp) > 1e-6, f"{name}: identical to arm {prev}, the load was a no-op"
        seen_fp[name] = got
        log(f"{name}: LoRA applied, |lora_B|={got:.3f} matches {os.path.basename(ckpt)}")

        adir = OUT / name
        t0 = time.time()
        saved = runner.run(model, step=0, output_dir=adir, device=device, progress=prog)
        log(f"{name}: {len(saved)} renders in {(time.time() - t0) / 60:.1f} min -> {adir}/samples")

        paths = {}
        for e in saved:
            if isinstance(e, (tuple, list)) and len(e) == 2:
                paths[int(e[0])] = e[1]
            else:
                paths[len(paths)] = e
        per, ms = {}, []
        for idx in range(len(samples)):
            p = paths.get(idx)
            png_dir = (str(Path(p).with_suffix("")) + "_png") if p is not None else \
                str(adir / "samples" / f"step_000000_{idx + 1}_png")
            m, why = _fg_iou(png_dir, gts[idx])
            per[os.path.basename(png_dir)] = {"iou": (round(m, 4) if m is not None else None),
                                              "why": why, "gt": os.path.basename(gts[idx])}
            if m is not None:
                ms.append(m)
        mean = round(float(np.mean(ms)), 4) if ms else None
        results["arms"][name] = {"lora_ckpt": ckpt, "mean_fg_iou": mean, "n_scored": len(ms),
                                 "n_samples": len(samples), "comparable": len(ms) == 8,
                                 "lora_b_abs_sum": round(got, 3), "per_sample": per}
        log(f"{name}: mean_fg_iou={mean} over {len(ms)}/8 samples")
        write_results()

# Verdict, pinned before launch: the control arm re-renders the phase-A step-750 LoRA and must
# return exactly the archived 0.5547. Anything else means this apparatus is not the one that
# produced the reference number, so the gate is UNSCORED rather than reported either way.
ctrl_name = spec["control_arm"]
prim_name = spec["primary_arm"]
ctrl, prim = results["arms"].get(ctrl_name), results["arms"].get(prim_name)
if not ctrl or ctrl["mean_fg_iou"] != REFERENCE or not ctrl["comparable"]:
    results["verdict"] = (f"G-SM2 UNSCORED: control arm returned {ctrl and ctrl['mean_fg_iou']} "
                          f"over {ctrl and ctrl['n_scored']}/8, expected {REFERENCE} over 8/8")
elif not prim or not prim["comparable"]:
    results["verdict"] = f"G-SM2 UNSCORED: primary arm scored {prim and prim['n_scored']}/8 samples"
else:
    verdict = "PASS" if REFERENCE > prim["mean_fg_iou"] else "FAIL"
    results["verdict"] = (f"G-SM2 {verdict}: phase-A {REFERENCE} vs zero-shot "
                          f"{prim['mean_fg_iou']} (control reproduced {ctrl['mean_fg_iou']})")
write_results()
PUB.mkdir(parents=True, exist_ok=True)
shutil.copytree(OUT, PUB, dirs_exist_ok=True)
log(results["verdict"])
log(f"VALPASS-DONE -> {PUB}")
