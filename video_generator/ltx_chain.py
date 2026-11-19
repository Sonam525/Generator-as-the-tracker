"""Multi-video windowed generation for BOTH full-video arms (run inside the trainer venv).

mode="chain":       window k conditioned on [IC-LoRA reference RGB] + [prefix = cleaned id-map tail
                    of window k-1] -- the decode->correct->re-encode loop; the generator carries
                    identity across boundaries (phase-3 checkpoint).
mode="independent": windows conditioned on [reference RGB] only (phase-2 checkpoint; post-hoc v3
                    stitcher applied at eval).

Loads Gemma + transformer ONCE for all videos/windows (runner constructed BEFORE the transformer for
VRAM). Per video: generate all windows -> clean each (codec decode->correct->repaint) -> publish that
video's outputs to the volume immediately (a crash only loses the in-flight video). Videos already
published (DONE.txt) are skipped, so relaunching resumes.

Usage: uv run python ltx_chain.py <chain_spec.json>
"""
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# --- lossless-PNG patch: must run before the trainer imports validation (same as ltx_train.py) ---
import ltx_trainer.video_utils as vu  # noqa: E402
from ltx_trainer.video_utils import _prepare_video_array  # noqa: E402

_orig_save_video = vu.save_video


def _save_video_png(video_tensor, output_path, fps=24.0, audio=None,
                    audio_sample_rate=None, video_format=None):
    arr = _prepare_video_array(video_tensor, video_format=video_format)  # [F,H,W,C] uint8
    d = Path(str(Path(output_path).with_suffix("")) + "_png")
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(arr):
        Image.fromarray(f.astype("uint8")).save(d / f"{i:04d}.png")
    try:
        _orig_save_video(video_tensor, output_path, fps, audio, audio_sample_rate, video_format)
    except Exception as e:
        print(f"[patch] mp4 save skipped: {e}", flush=True)


vu.save_video = _save_video_png
import ltx_trainer.validation_runner as _vr  # noqa: E402
_vr.save_video = _save_video_png

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from ltx_trainer.config import ValidationConfig  # noqa: E402
from ltx_trainer.model_loader import load_transformer  # noqa: E402
from ltx_trainer.progress import TrainingProgress  # noqa: E402
from ltx_trainer.validation_runner import ValidationRunner  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [chain] {m}", flush=True)


spec = json.load(open(sys.argv[1]))
sys.path.insert(0, spec["gentrack_src"])
from gentrack.color_codec import decode as codec_decode  # noqa: E402
from gentrack.palette import build  # noqa: E402

MODE = spec["mode"]                                   # "chain" | "independent"
K, MIN_AREA, PREFIX = spec["K"], spec["min_area"], spec["prefix_frames"]
palette = np.array(build(K), dtype=np.uint8)
OUT = Path(spec["output_dir"])
PUB = Path(spec["publish_root"])
VIDEOS = spec["videos"]                               # list of vid names
WROOT = spec["windows_root"]                          # {VOL}/ltx_ds/fullseq
SUFFIX = spec["win_suffix"]                           # _ov9
log(f"mode={MODE} | {len(VIDEOS)} videos | ckpt={Path(spec['lora_checkpoint']).name}")


def make_cfg(vid, wins, k, prefix_dir):
    conds = [dict(type="reference", video=f"{WROOT}/{vid}{SUFFIX}/rgb/{wins[k]['tag']}.mkv",
                  downscale_factor=1, temporal_scale_factor=1, include_in_output=True)]
    if MODE == "chain" and k > 0:
        conds.append(dict(type="prefix", video=str(prefix_dir / f"w{k:02d}.mkv"),
                          num_frames=PREFIX))
    return ValidationConfig.model_validate(dict(
        samples=[dict(prompt=spec["prompt"], conditions=conds)],
        negative_prompt=spec["negative_prompt"],
        video_dims=spec["video_dims"], frame_rate=24.0, seed=spec["seed"],
        inference_steps=spec["inference_steps"], interval=100000,
        guidance_scale=spec["guidance_scale"], stg_scale=spec["stg_scale"],
        stg_blocks=spec["stg_blocks"], stg_mode=spec["stg_mode"],
        generate_audio=False, skip_initial_validation=False,
    ))


def clean_frame(rgb):
    inst = codec_decode(rgb, K=K, min_area=MIN_AREA)
    out = np.zeros_like(rgb)
    for cidx, mask in inst.items():
        out[mask] = palette[cidx]
    return out


def right_half_frames(d):
    out = []
    for p in sorted(glob.glob(f"{d}/*.png")):
        f = np.asarray(Image.open(p))[:, :, :3]
        if f.shape[1] == 2 * f.shape[0]:
            f = f[:, f.shape[1] // 2:, :]
        out.append(f)
    return out


def write_lossless(path, frames):
    w = imageio.get_writer(str(path), fps=24, codec="ffv1", macro_block_size=1,
                           format="FFMPEG", output_params=["-pix_fmt", "rgb24"])
    for fr in frames:
        w.append_data(fr)
    w.close()


todo = [v for v in VIDEOS if not (PUB / v / "DONE.txt").exists()]
log(f"resume check: {len(VIDEOS) - len(todo)} already published, {len(todo)} to generate")
if not todo:
    log("nothing to do"); sys.exit(0)

# first video's first window config bootstraps the runner (Gemma load -> cache -> unload)
meta0 = json.load(open(f"{WROOT}/{todo[0]}{SUFFIX}/windows.json"))
wins0 = sorted(meta0["windows"], key=lambda w: w["idx"])
t0 = time.time()
runner = ValidationRunner(config=make_cfg(todo[0], wins0, 0, OUT / todo[0] / "prefix"),
                          model_path=spec["model_path"],
                          text_encoder_path=spec["text_encoder_path"],
                          load_text_encoder_in_8bit=False)
log(f"runner ready in {time.time()-t0:.0f}s")
t0 = time.time()
transformer = load_transformer(spec["model_path"], device="cuda", dtype=torch.bfloat16)
model = get_peft_model(transformer, LoraConfig(
    r=spec["lora_rank"], lora_alpha=spec["lora_alpha"],
    target_modules=spec["lora_targets"], lora_dropout=0.0, init_lora_weights=True))
sd = load_file(spec["lora_checkpoint"])
sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
set_peft_model_state_dict(model.get_base_model(), sd)
model.eval()
log(f"transformer + LoRA ready in {time.time()-t0:.0f}s")

device = torch.device("cuda")
first = True
with TrainingProgress(enabled=False, total_steps=1) as prog:
    for vid in todo:
        meta = json.load(open(f"{WROOT}/{vid}{SUFFIX}/windows.json"))
        wins = sorted(meta["windows"], key=lambda w: w["idx"])
        if MODE == "chain":
            assert meta["overlap"] == PREFIX, f"{vid}: overlap {meta['overlap']} != prefix {PREFIX}"
        vout = OUT / vid
        (vout / "prefix").mkdir(parents=True, exist_ok=True)
        (vout / "cleaned").mkdir(parents=True, exist_ok=True)
        timings = []
        tvid = time.time()
        for k in range(len(wins)):
            if not first:
                runner._config = make_cfg(vid, wins, k, vout / "prefix")
                runner._cached_media = runner._encode_conditioning_media()
            first = False
            tw = time.time()
            runner.run(model, step=k, output_dir=vout, device=device, progress=prog)
            gen_s = time.time() - tw
            png_dir = vout / "samples" / f"step_{k:06d}_1_png"
            frames = right_half_frames(str(png_dir))
            assert frames, f"{vid} window {k}: no generated frames at {png_dir}"
            cleaned = [clean_frame(f) for f in frames]
            cdir = vout / "cleaned" / f"w{k:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(cleaned):
                Image.fromarray(f).save(cdir / f"{i:04d}.png")
            if MODE == "chain" and k + 1 < len(wins):
                write_lossless(vout / "prefix" / f"w{k+1:02d}.mkv", cleaned[-PREFIX:])
            timings.append(dict(window=k, gen_s=round(gen_s, 1), frames=len(frames)))
            with open(vout / "chain_log.json", "w") as fh:
                json.dump(timings, fh)
        # incremental publish: this video is durable before the next starts
        dest = PUB / vid
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vout / "cleaned", dest / "cleaned", dirs_exist_ok=True)
        shutil.copy(vout / "chain_log.json", dest / "chain_log.json")
        (dest / "DONE.txt").write_text(f"{len(wins)} windows, {time.time()-tvid:.0f}s\n")
        shutil.rmtree(vout / "samples", ignore_errors=True)   # keep local disk lean
        log(f"{vid}: {len(wins)} windows in {(time.time()-tvid)/60:.1f} min "
            f"(mean {np.mean([t['gen_s'] for t in timings]):.0f}s/window) -> published")

log(f"ALL-DONE mode={MODE} videos={len(todo)}")
