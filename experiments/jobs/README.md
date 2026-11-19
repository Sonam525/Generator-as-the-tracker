# Job records

Databricks job definitions as they were submitted. They are records, not templates: notebook
paths are the ones in use at the time (the flat `scripts/` layout, see `docs/PATH_MAP.md`).

- `paper_runs/`: the SportsMOT runs, the factorial cells, the seed runs and the review-stage
  evaluations, with their parameters. User names are replaced by `<user>`.
- `gpu_diag.py`: a cluster diagnostic (GPU visibility, scratch-disk permissions).

The launch files of the June 2026 image-generator phase held only run names, cluster ids and
notebook paths, no parameters; they were removed on 2026-11-19 and are kept privately.

No job definition survives for the DanceTrack phase A and phase B training runs, the first
arm A and arm B generations, or the test-set chain; their settings are in
`video_generator/configs/` and in Appendix A of the paper.
