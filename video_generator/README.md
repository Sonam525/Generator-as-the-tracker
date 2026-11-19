# Video generator

The code behind the paper: LTX-2.3 fine-tuned with an in-context LoRA to translate an RGB clip
into an ID-map clip, chained over full videos, and evaluated.

Scripts come in three kinds. Databricks notebooks (every file that starts with
`# Databricks notebook source`) run on the cluster driver. Workers (`ltx_train.py`,
`ltx_chain.py`, `ltx_valpass.py`, `ltx_density_probe.py`) run inside the LTX-2 trainer's own
environment and are launched by their runner notebook, which looks for them at
`$GENTRACK_WS/video_generator/`, so they must stay in this folder. `make_sportsmot_configs.py`
is a local command-line script, run from the repository root.

## Pipeline, in order

| Step | File | What it does |
|---|---|---|
| Weights | `download_ltx_native.py` | LTX-2.3 weights and the text encoder to the volume |
| Training pairs | `ltx_make_dataset.py` | Aligned RGB and ID-map clips as lossless video |
| Train | `ltx_run.py` + `ltx_train.py` | Runs the LTX trainer with a config from `configs/`, with lossless validation and early stopping |
| Windows | `ltx_render_windows.py` | Overlapping 49-frame RGB windows for each validation video |
| Generate | `ltx_run_chain.py` + `ltx_chain.py` | Full-video generation, chained or as independent windows; takes a seed |
| Evaluate | `ltx_arm_eval.py` | Decodes windows to tracks and runs TrackEval over the 25 validation videos |
| Re-acquisition | `ltx_reacq_study.py` | First re-acquisition study: mines occlusion gaps and scores both arms. The paper's re-acquisition table uses the corrected protocol in `ltx_assoc_baselines.py` (arm B scored on union-merged boxes) |
| Baselines and controls | `ltx_assoc_baselines.py` | Kalman/IoU and DINOv3-bank association on the same boxes, re-acquisition and carry-over events; produced the records behind the baseline, re-acquisition, factorial, seed and SportsMOT tables (the per-video and sub-metric tables read `ltx_arm_eval.py` records) |
| Test server | `ltx_render_test.py`, `ltx_test_submit.py` | Test-split windows and the submission archive |

## Second benchmark and review-stage probes

| File | What it does |
|---|---|
| `make_sportsmot_configs.py` | Derives the two SportsMOT training configs from the DanceTrack ones |
| `ltx_valpass_run.py` + `ltx_valpass.py` | Scores checkpoints on the SportsMOT selection windows with zero training steps |
| `ltx_density_probe_run.py` + `ltx_density_probe.py` | Renders training clips of different crowd density and counts painted regions |
| `ltx_size_recall.py` | Detection recall by person size and merge rate on dense videos |

## `configs/`

Trainer configs, plus `ltx_vid.yaml`, the config of the abandoned diffusers-format attempt
(see `ablations/`). File names are kept as they were during the runs, because each name is also
the name of its checkpoint and result folder on the volume. In the paper, `phase2` is phase A
(translation) and `phase3` is phase B (continuation); `sm_` marks SportsMOT.

## `ablations/`

Development studies, mostly on one video (dancetrack0065): the window-stitching series
(`ltx_stitch_*`; `ltx_stitch_v3.py` introduced the many-to-one linker that `ltx_arm_eval.py`
and `ltx_assoc_baselines.py` re-implement as link mode `v3many` for independent windows), the
121-frame length test (`ltx_render_one.py`, `ltx_length_eval.py`), single-window evaluation
(`ltx_track_eval.py`, `ltx_hota_eval.py`, and `ltx_p1_eval.py` on three held-out videos across
checkpoints), the first run on the native LTX trainer (`ltx_train_phase1.py`, a 12-clip
sanity run), and the earlier attempt on the diffusers-format model (`download_ltx.py`,
`ltx_probe.py`, `ltx_cond_sanity.py`).

## `submission/`

The frozen DanceTrack test-server entry, its manifest and the scores the server returned.

## Arm names in the records

| Record name | Checkpoint | Generation |
|---|---|---|
| arm A | phase 3 (continuation-trained) | chained |
| arm B | phase 2 (translation only) | independent windows, linked afterwards |
| arm C | phase 2 | chained |
| arm D | phase 3 | independent windows |
