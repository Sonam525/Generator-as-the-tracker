# Earlier trials with image generators

Before the video generator, the project tried to make *image* generators carry identity from
frame to frame. Two backbones were tried. Neither is part of the paper's method; this folder
keeps the code as it last ran, as the record behind the June 2026 runs. `FINDINGS.md` is the write-up of what these trials concluded
(dated 2026-06-18, before the video arm reversed its main negative result).

Several scripts were edited in place between runs, so each file shows the last configuration
that was run, not every variant in the log.

The Cosmos LoRA weights and the Cosmos3-Nano base model were deleted from the volume on
2026-11-19 (the training logs and sample images were kept). Re-running the Cosmos scripts
needs the base model downloaded again and the LoRAs retrained (about 1.5 A100-hours each).

## `nano_banana/` : Gemini image API with a memory canvas ("Option A", zero training)

Local scripts that call a paid API (`GEMINI_API_KEY` from the environment). Output folders
default to `pa_scratch/<name>` under the current folder (git-ignored); pass `<out_dir>` where the script accepts one,
or edit `OUT` in `pa_inspect.py` and `pa_diag2.py`. The adapter,
canvas and tracker live in `core/gentrack` (`models/nanobanana_api.py`,
`pipeline/memory_canvas.py`, `pipeline/track.py`, `eval/run_pA.py`).

| File | What it does |
|---|---|
| `run_pA_canvas.sh` | Driver for the memory-canvas baseline on one sequence |
| `smoke_nb2.py` | One-image contract test of the API on a synthetic frame |
| `pa_inspect.py` | Inspects raw ID-map colours on real frames and tests colour propagation |
| `pa_diag2.py` | Instrumented 10-frame loop, logs per-frame matching |
| `pa_rerun.py` | Runs a clip through the canvas loop and scores it (the logged baseline) |
| `pa_reacq.py` | Occlusion re-acquisition test on a mined disappear/reappear clip |
| `pa_eval.py` | Re-scores saved maps against ground truth without calling the API |
| `nanobanana_api.yaml`, `eval.yaml` | Model id, prompts, retry policy, spend gate; decode and persistence parameters |

## `cosmos/` : Cosmos3-Nano fine-tuning in image mode

Databricks notebooks (one A100; the Cosmos3-Super fit test used two). Adapter and identity
bank are in `core/gentrack/models`.

| File | What it does |
|---|---|
| `download_cosmos.py`, `calibrate_cosmos.py` | Download and fit test of the larger Cosmos3-Super model (did not fit) |
| `setup_cosmos_nano.py` | Download and calibration of Cosmos3-Nano |
| `cond_probe.py`, `cond_sanity.py` | How the model takes an image condition; overfit proof that it uses the frame |
| `p0_finetune.py` | Stage-1 LoRA trainer: frame to ID-map |
| `p1_finetune.py` | Stage-2 temporal LoRA trainer: frame plus previous map to ID-map |
| `eval_probe.py`, `eval_trained.py`, `p2_temporal_eval.py` | Chained identity evaluation of the trained LoRAs |
| `p2_feasibility.py` | No-state baseline: per-frame perception with heuristic association |
| `inject_sanity.py`, `xattn_sanity.py` | Embedding-bank injection tests (negative) |
| `p2_reid_eval.py`, `reacq_diag.py` | Re-identification on stable perception; appearance diagnostic |
| `mot17_perceive_probe.py`, `mot17_reid_eval.py` | Zero-shot transfer and re-identification on MOT17 |
| `train.yaml` | Stage-1 LoRA hyperparameters |
