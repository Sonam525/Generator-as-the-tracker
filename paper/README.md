# Paper

Two versions of one paper share the same figures, tables and bibliography, so their numbers
cannot drift apart.

| Path | Contents |
|---|---|
| `iclr2027/` | The ICLR 2027 version: anonymous, revision marks in blue. `gentrack_iclr.tex` and its PDF |
| `arxiv/` | The arXiv version ([arXiv:2607.17120](https://arxiv.org/abs/2607.17120)), derived from `iclr2027/` by `scripts/make_arxiv.py`: authors shown, header "Preprint", no blue marks, ethics statement reworded for the dataset frames and videos shown for reference, reproducibility statement pointing to this repository instead of the anonymised supplement |
| `figures/`, `tables/` | Built by the scripts below; both versions include them |
| `references.bib` | The bibliography of both versions |
| `scripts/` | The builders listed below |
| `data/` | Run records behind every table: copies of the volume's `results/<run>` outputs (folders keep the run name, older single files are flattened to `<run>_<file>`), plus the `*_merged` folders built by `make_revision_tables.py` |
| `assets/` | Frames, ID-maps and event lists used by the figures |

## Scripts

| Script | What it builds |
|---|---|
| `make_tables.py`, `make_figures.py` | Per-video, sub-metric and re-acquisition tables; all figures |
| `make_revision_tables.py` | Copies run records from the volume (Databricks CLI, volume path set in the script), merges them and draws the chain-length figure |
| `make_sportsmot_table.py`, `make_seeds_table.py` | SportsMOT and seed tables |
| `make_rebuttal_analysis.py`, `make_rebuttal_stats.py` | Factorial, seed and confidence-interval numbers |
| `density_probe_exploratory.py`, `painted_per_frame_exploratory.py` | Post-hoc analyses, labelled exploratory in the paper |
| `make_arxiv.py` | Writes `arxiv/` from `iclr2027/`; the list of differences is at the top of the script |
| `pack_arxiv.py` | Packs `arxiv/` with the figures, tables and `.bbl` into `arxiv/gentrack_arxiv_upload.zip` and test-builds it as arXiv does |

All scripts use paths relative to the repository root and must be run from there.

## Updating the arXiv version

Edit the ICLR source (or the shared figures and tables), then:

```bash
python paper/scripts/make_arxiv.py
cd paper/arxiv && latexmk -pdf gentrack_arxiv.tex && cd ../..
python paper/scripts/pack_arxiv.py
```

Upload `paper/arxiv/gentrack_arxiv_upload.zip` on arXiv as a replacement of 2607.17120. The zip
is git-ignored.

Earlier drafts and submission material are not part of this repository.
