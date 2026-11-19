# Path map: layout before and after 2026-11-19

The repository was reorganised into area folders on 2026-11-19. The experiment log (kept
privately), the job records under `experiments/jobs/`, the run records under
`paper/data/`, `image_generators/FINDINGS.md`, the original project plan (kept privately), the frozen files
in `src/gentrack` (now `core/gentrack`) and the anonymised supplement (now kept privately) were written before that date and cite the
old paths. They are left as written; this table translates them. Files removed after the move
are listed at the end.

No file content changed in the move except path lines: the workspace folder is now read from
the `GENTRACK_WS` environment variable, the volume root from `GENTRACK_VOL` (default: the
original volume path), and references between scripts and configs point at the new locations.

## Scripts

| Before | After |
|---|---|
| `scripts/calibrate_cosmos.py` | `image_generators/cosmos/calibrate_cosmos.py` |
| `scripts/cond_probe.py` | `image_generators/cosmos/cond_probe.py` |
| `scripts/cond_sanity.py` | `image_generators/cosmos/cond_sanity.py` |
| `scripts/download_cosmos.py` | `image_generators/cosmos/download_cosmos.py` |
| `scripts/download_dancetrack.py` | `datasets/download_dancetrack.py` |
| `scripts/download_ltx.py` | `video_generator/ablations/download_ltx.py` |
| `scripts/download_ltx_native.py` | `video_generator/download_ltx_native.py` |
| `scripts/download_mot17.py` | `datasets/download_mot17.py` |
| `scripts/download_sportsmot.py` | `datasets/download_sportsmot.py` |
| `scripts/eval_probe.py` | `image_generators/cosmos/eval_probe.py` |
| `scripts/eval_trained.py` | `image_generators/cosmos/eval_trained.py` |
| `scripts/find_mot17.py` | `datasets/find_mot17.py` |
| `scripts/gpu_diag.py` | `experiments/jobs/gpu_diag.py` |
| `scripts/inject_probe.py` | `image_generators/cosmos/inject_probe.py` |
| `scripts/inject_sanity.py` | `image_generators/cosmos/inject_sanity.py` |
| `scripts/ltx_arm_eval.py` | `video_generator/ltx_arm_eval.py` |
| `scripts/ltx_assoc_baselines.py` | `video_generator/ltx_assoc_baselines.py` |
| `scripts/ltx_chain.py` | `video_generator/ltx_chain.py` |
| `scripts/ltx_cond_sanity.py` | `video_generator/ablations/ltx_cond_sanity.py` |
| `scripts/ltx_density_probe.py` | `video_generator/ltx_density_probe.py` |
| `scripts/ltx_density_probe_run.py` | `video_generator/ltx_density_probe_run.py` |
| `scripts/ltx_hota_eval.py` | `video_generator/ablations/ltx_hota_eval.py` |
| `scripts/ltx_length_eval.py` | `video_generator/ablations/ltx_length_eval.py` |
| `scripts/ltx_make_dataset.py` | `video_generator/ltx_make_dataset.py` |
| `scripts/ltx_p1_eval.py` | `video_generator/ablations/ltx_p1_eval.py` |
| `scripts/ltx_probe.py` | `video_generator/ablations/ltx_probe.py` |
| `scripts/ltx_reacq_study.py` | `video_generator/ltx_reacq_study.py` |
| `scripts/ltx_render_one.py` | `video_generator/ablations/ltx_render_one.py` |
| `scripts/ltx_render_test.py` | `video_generator/ltx_render_test.py` |
| `scripts/ltx_render_windows.py` | `video_generator/ltx_render_windows.py` |
| `scripts/ltx_run.py` | `video_generator/ltx_run.py` |
| `scripts/ltx_run_chain.py` | `video_generator/ltx_run_chain.py` |
| `scripts/ltx_size_recall.py` | `video_generator/ltx_size_recall.py` |
| `scripts/ltx_stitch_chain_eval.py` | `video_generator/ablations/ltx_stitch_chain_eval.py` |
| `scripts/ltx_stitch_eval.py` | `video_generator/ablations/ltx_stitch_eval.py` |
| `scripts/ltx_stitch_eval2.py` | `video_generator/ablations/ltx_stitch_eval2.py` |
| `scripts/ltx_stitch_oracle.py` | `video_generator/ablations/ltx_stitch_oracle.py` |
| `scripts/ltx_stitch_v3.py` | `video_generator/ablations/ltx_stitch_v3.py` |
| `scripts/ltx_stitch_v4.py` | `video_generator/ablations/ltx_stitch_v4.py` |
| `scripts/ltx_stitch_v5.py` | `video_generator/ablations/ltx_stitch_v5.py` |
| `scripts/ltx_stitch_wc.py` | `video_generator/ablations/ltx_stitch_wc.py` |
| `scripts/ltx_test_submit.py` | `video_generator/ltx_test_submit.py` |
| `scripts/ltx_track_eval.py` | `video_generator/ablations/ltx_track_eval.py` |
| `scripts/ltx_train.py` | `video_generator/ltx_train.py` |
| `scripts/ltx_train_phase1.py` | `video_generator/ablations/ltx_train_phase1.py` |
| `scripts/ltx_valpass.py` | `video_generator/ltx_valpass.py` |
| `scripts/ltx_valpass_run.py` | `video_generator/ltx_valpass_run.py` |
| `scripts/make_sportsmot_configs.py` | `video_generator/make_sportsmot_configs.py` |
| `scripts/mine_occlusion_val.py` | `datasets/mine_occlusion_val.py` |
| `scripts/mot17_perceive_probe.py` | `image_generators/cosmos/mot17_perceive_probe.py` |
| `scripts/mot17_reid_eval.py` | `image_generators/cosmos/mot17_reid_eval.py` |
| `scripts/p0_finetune.py` | `image_generators/cosmos/p0_finetune.py` |
| `scripts/p0_introspect.py` | `image_generators/cosmos/p0_introspect.py` |
| `scripts/p0_nframes_probe.py` | `image_generators/cosmos/p0_nframes_probe.py` |
| `scripts/p0_train_probe.py` | `image_generators/cosmos/p0_train_probe.py` |
| `scripts/p0_train_step.py` | `image_generators/cosmos/p0_train_step.py` |
| `scripts/p1_finetune.py` | `image_generators/cosmos/p1_finetune.py` |
| `scripts/p2_feasibility.py` | `image_generators/cosmos/p2_feasibility.py` |
| `scripts/p2_reid_eval.py` | `image_generators/cosmos/p2_reid_eval.py` |
| `scripts/p2_temporal_eval.py` | `image_generators/cosmos/p2_temporal_eval.py` |
| `scripts/pa_diag.py` | `image_generators/nano_banana/pa_diag.py` |
| `scripts/pa_diag2.py` | `image_generators/nano_banana/pa_diag2.py` |
| `scripts/pa_eval.py` | `image_generators/nano_banana/pa_eval.py` |
| `scripts/pa_inspect.py` | `image_generators/nano_banana/pa_inspect.py` |
| `scripts/pa_reacq.py` | `image_generators/nano_banana/pa_reacq.py` |
| `scripts/pa_rerun.py` | `image_generators/nano_banana/pa_rerun.py` |
| `scripts/pa_validate.py` | `image_generators/nano_banana/pa_validate.py` |
| `scripts/reacq_diag.py` | `image_generators/cosmos/reacq_diag.py` |
| `scripts/run_pA_canvas.sh` | `image_generators/nano_banana/run_pA_canvas.sh` |
| `scripts/sam3_pseudo_masks.py` | `datasets/sam3_pseudo_masks.py` |
| `scripts/sam3_smoke.py` | `datasets/sam3_smoke.py` |
| `scripts/setup_cosmos_nano.py` | `image_generators/cosmos/setup_cosmos_nano.py` |
| `scripts/smoke_nb2.py` | `image_generators/nano_banana/smoke_nb2.py` |
| `scripts/xattn_sanity.py` | `image_generators/cosmos/xattn_sanity.py` |

## Configs

| Before | After |
|---|---|
| `configs/data.yaml` | `datasets/data.yaml` |
| `configs/eval.yaml` | `image_generators/nano_banana/eval.yaml` |
| `configs/ltx_v2v_ic_lora_diag.yaml` | `video_generator/configs/ltx_v2v_ic_lora_diag.yaml` |
| `configs/ltx_v2v_ic_lora_eval.yaml` | `video_generator/configs/ltx_v2v_ic_lora_eval.yaml` |
| `configs/ltx_v2v_ic_lora_fullseq.yaml` | `video_generator/configs/ltx_v2v_ic_lora_fullseq.yaml` |
| `configs/ltx_v2v_ic_lora_fullseq_p3.yaml` | `video_generator/configs/ltx_v2v_ic_lora_fullseq_p3.yaml` |
| `configs/ltx_v2v_ic_lora_infercal.yaml` | `video_generator/configs/ltx_v2v_ic_lora_infercal.yaml` |
| `configs/ltx_v2v_ic_lora_lengthtest.yaml` | `video_generator/configs/ltx_v2v_ic_lora_lengthtest.yaml` |
| `configs/ltx_v2v_ic_lora_phase1.yaml` | `video_generator/configs/ltx_v2v_ic_lora_phase1.yaml` |
| `configs/ltx_v2v_ic_lora_phase2.yaml` | `video_generator/configs/ltx_v2v_ic_lora_phase2.yaml` |
| `configs/ltx_v2v_ic_lora_phase3.yaml` | `video_generator/configs/ltx_v2v_ic_lora_phase3.yaml` |
| `configs/ltx_v2v_ic_lora_sm_phase2.yaml` | `video_generator/configs/ltx_v2v_ic_lora_sm_phase2.yaml` |
| `configs/ltx_v2v_ic_lora_sm_phase3.yaml` | `video_generator/configs/ltx_v2v_ic_lora_sm_phase3.yaml` |
| `configs/ltx_vid.yaml` | `video_generator/configs/ltx_vid.yaml` |
| `configs/nanobanana_api.yaml` | `image_generators/nano_banana/nanobanana_api.yaml` |
| `configs/train.yaml` | `image_generators/cosmos/train.yaml` |

## Root files

| Before | After |
|---|---|
| `FINDINGS.md` | `image_generators/FINDINGS.md` |
| `a10_cluster_spec.json` | `experiments/jobs/image_phase/a10_cluster_spec.json` |
| `cal_run.json` | `experiments/jobs/image_phase/cal_run.json` |
| `cond_run.json` | `experiments/jobs/image_phase/cond_run.json` |
| `cosmos_run.json` | `experiments/jobs/image_phase/cosmos_run.json` |
| `download_run.json` | `experiments/jobs/image_phase/download_run.json` |
| `ev_run.json` | `experiments/jobs/image_phase/ev_run.json` |
| `evprobe_run.json` | `experiments/jobs/image_phase/evprobe_run.json` |
| `feas_run.json` | `experiments/jobs/image_phase/feas_run.json` |
| `ft_run.json` | `experiments/jobs/image_phase/ft_run.json` |
| `generative-tracking-plan-v2.md` | kept privately by the authors |
| `inject_run.json` | `experiments/jobs/image_phase/inject_run.json` |
| `injsanity_run.json` | `experiments/jobs/image_phase/injsanity_run.json` |
| `introspect_run.json` | `experiments/jobs/image_phase/introspect_run.json` |
| `mot17dl_run.json` | `experiments/jobs/image_phase/mot17dl_run.json` |
| `mot17find_run.json` | `experiments/jobs/image_phase/mot17find_run.json` |
| `mot17pp_run.json` | `experiments/jobs/image_phase/mot17pp_run.json` |
| `mot17reid_run.json` | `experiments/jobs/image_phase/mot17reid_run.json` |
| `nano_run.json` | `experiments/jobs/image_phase/nano_run.json` |
| `nf_run.json` | `experiments/jobs/image_phase/nf_run.json` |
| `p0sam_run.json` | `experiments/jobs/image_phase/p0sam_run.json` |
| `p1_run.json` | `experiments/jobs/image_phase/p1_run.json` |
| `p2reid_run.json` | `experiments/jobs/image_phase/p2reid_run.json` |
| `p2temporal_run.json` | `experiments/jobs/image_phase/p2temporal_run.json` |
| `p2teval_run.json` | `experiments/jobs/image_phase/p2teval_run.json` |
| `probe_run.json` | `experiments/jobs/image_phase/probe_run.json` |
| `reacqdiag_run.json` | `experiments/jobs/image_phase/reacqdiag_run.json` |
| `sam3_run.json` | `experiments/jobs/image_phase/sam3_run.json` |
| `sam3pass_run.json` | `experiments/jobs/image_phase/sam3pass_run.json` |
| `sanity_run.json` | `experiments/jobs/image_phase/sanity_run.json` |
| `ts_run.json` | `experiments/jobs/image_phase/ts_run.json` |
| `xattn_run.json` | `experiments/jobs/image_phase/xattn_run.json` |

## Benchmark submission

| Before | After |
|---|---|
| `submission/` (whole folder) | `video_generator/submission/` |

## Paper

| Before | After |
|---|---|
| `paper/iclr2026/` (whole folder) | `paper/archive/iclr2026/` |
| `paper/arxiv/` (whole folder) | `paper/archive/arxiv/` |
| `paper/audit_tex.py` | `paper/archive/one_off_scripts/audit_tex.py` |
| `paper/fix_table_rows.py` | `paper/archive/one_off_scripts/fix_table_rows.py` |
| `paper/migrate_2027.py` | `paper/archive/one_off_scripts/migrate_2027.py` |
| `paper/patch_arxiv.py` | `paper/archive/one_off_scripts/patch_arxiv.py` |
| `paper/patch_calibrate.py` | `paper/archive/one_off_scripts/patch_calibrate.py` |
| `paper/patch_conclusion.py` | `paper/archive/one_off_scripts/patch_conclusion.py` |
| `paper/sync_arxiv_ai.py` | `paper/archive/one_off_scripts/sync_arxiv_ai.py` |

## Not moved in the first move

`src/gentrack/`, `tests/`, `conftest.py`, `pyproject.toml`,
`paper/iclr2027/`, `paper/data/`, `paper/assets/` and the table and figure scripts in `paper/`.
The second move below then moved the library, the tests and the paper scripts.

Job files that the log cites as `_dl/<name>.json` (a local folder that was never tracked) are
in `experiments/jobs/paper_runs/` under the same name.

On the Databricks workspace, notebooks were addressed without the `.py` extension, for example
`.../Generative Tracker/scripts/ltx_run` is now `video_generator/ltx_run`.

## Removed after the move (2026-11-19)

Some targets in the tables above no longer exist in the repository.

Deleted (superseded, duplicated or unused; no table, figure or reported result is computed by
them, though the private experiment log still itemises the compute of some of their runs):
`image_generators/nano_banana/pa_validate.py`, `pa_diag.py`; `datasets/find_mot17.py`;
`image_generators/cosmos/p0_introspect.py`, `p0_train_probe.py`, `p0_nframes_probe.py`,
`p0_train_step.py`, `inject_probe.py`; `paper/archive/one_off_scripts/`;
`paper/data/ltx_reb_mech_A_merged/`, `ltx_seeds_eval_merged/` (rebuilt by
`make_revision_tables.py`), `ltx_revision_smoke/` (the run's copy stays on the volume under
`results/`); `video_generator/submission/gentrack_ltx_test_submission.zip` (same contents as
`tracker.zip`); the six `paper/iclr2027/figures/fig_chain_decay_ltx_*.pdf` by-products; unused
figure candidates in `paper/assets/`.

Kept privately by the authors, not in the repository: `paper/archive/arxiv/`,
`paper/archive/iclr2026/`, `paper/supplementary/`, other submission material,
`paper/make_supplement.py`, `experiments/jobs/image_phase/`.

## Second move (2026-11-19, later the same day)

| Before | Now |
|---|---|
| `src/gentrack/` | `core/gentrack/` (the `src/` level was dropped; notebooks now add `$GENTRACK_WS/core` to `sys.path`, and the chain and density-probe runners pass it to their workers as `gentrack_src`) |
| `tests/` | `core/tests/` |
| `conftest.py` | `core/conftest.py` |
| `paper/make_*.py`, `paper/*_exploratory.py` | `paper/scripts/` (same file names) |
| `paper/iclr2027/figures/`, `paper/iclr2027/tables/` | `paper/figures/`, `paper/tables/` (shared by both paper versions) |
| `paper/iclr2027/gentrack_refs.bib` | `paper/references.bib` |
| (new) | `paper/arxiv/`, the arXiv v2 source, with `paper/scripts/make_arxiv.py` and `pack_arxiv.py`. Not the same folder as the earlier `paper/arxiv/` (the v1 source, now kept privately): older citations of `paper/arxiv/` refer to v1 |
| `.pytest_cache/`, `.ruff_cache/` | `.cache/pytest/`, `.cache/ruff/` (set in `pyproject.toml`) |
