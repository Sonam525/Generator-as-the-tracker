# Datasets

No dataset is stored in this repository. The only dataset material is a handful of DanceTrack
frames and annotation files under `paper/assets`, used by the paper's figures, and the
project-page videos in `docs/media/` (one full validation video and one excerpt), shown for
reference and credited to the DanceTrack authors. These scripts download each dataset to a mounted volume
and build the labels the generators are trained on. Loaders and label builders are in
`core/gentrack/data`.

| Dataset | Used for | Licence note |
|---|---|---|
| DanceTrack | Main benchmark: 40 train, 25 validation, 35 test videos | Non-commercial research use |
| SportsMOT | Second benchmark: train and validation splits | Non-commercial research use |
| MOT17 | Early transfer trials only (not in the paper) | Non-commercial (MOTChallenge terms); downloaded from a community mirror |

| File | What it does |
|---|---|
| `data.yaml` | Volume root, dataset roots and splits, palette size, mask source, mining gaps |
| `download_dancetrack.py` | DanceTrack train and validation archives to the volume; the test archives only when named in the `files` widget |
| `download_sportsmot.py` | SportsMOT train and validation, stored in MOT17 layout |
| `download_mot17.py` | MOT17 train split |
| `sam3_pseudo_masks.py` | SAM 3 silhouette labels over the train splits only (one instance map per frame) |
| `sam3_smoke.py` | Three-frame SAM 3 smoke test |
| `mine_occlusion_val.py` | Lists disappear/reappear gaps in ground truth, used to pick occlusion clips |

Validation and test splits never enter training in any form: pseudo-masks are generated for
train splits only.

The download and SAM 3 scripts are Databricks notebooks. They read the volume root from the
`GENTRACK_VOL` environment variable; the SAM 3 scripts also import `gentrack` and need
`GENTRACK_WS` (see the top-level README).
