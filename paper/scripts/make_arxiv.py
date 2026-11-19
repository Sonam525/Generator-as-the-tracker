"""Derive the arXiv source (paper/arxiv/) from the ICLR source (paper/iclr2027/).

    python paper/scripts/make_arxiv.py

The two versions share paper/figures, paper/tables and paper/references.bib, so the numbers
cannot drift apart. The arXiv version differs only by the edits listed in EDITS: author
block, PDF metadata, preprint header, release claims (the experiment log stays private), revision marks shown as plain text, and the ethics and
reproducibility statements (dataset frames and videos shown for reference; code at the public
repository instead of the anonymised supplement). Every edit must match exactly once, or the script stops.
"""
import shutil
from pathlib import Path

SRC, DST = Path("paper/iclr2027"), Path("paper/arxiv")
REPO = "https://github.com/Sonam525/Generator-as-the-tracker"
PAGE = "https://sonam525.github.io/Generator-as-the-tracker/"
HF = "https://huggingface.co/Sonam5/Generator-as-the-tracker"
STYLE = ["iclr2027_conference.bst", "natbib.sty", "fancyhdr.sty", "math_commands.tex"]

EDITS = [
    (r"\documentclass{article}", "\\pdfoutput=1  % arXiv: force pdflatex\n\\documentclass{article}"),
    ("% Rebuttal revision (Nov 2026): changed text in blue. Set to #1 for camera-ready.\n"
     r"\newcommand{\rev}[1]{{\color{blue}#1}}" "\n"
     r"\newcommand{\PEND}[1]{{\color{red}\textbf{[#1]}}} % must be gone before posting",
     "% Revision marks of the ICLR source, shown as plain text.\n"
     r"\newcommand{\rev}[1]{#1}"),
    ("% Anonymous for double-blind review (ICLR 2027: identity leak = desk reject)\n"
     "\\author{Anonymous authors\\\\\nPaper under double-blind review}",
     "\\author{Haiyu Yang \\quad Miel Hostens \\\\\nCornell University \\\\\n"
     "\\texttt{hy625@cornell.edu}\n}"),
    ("% \\iclrfinalcopy  % ENABLE ONLY FOR CAMERA-READY; submission build must stay anonymous",
     "\\iclrfinalcopy  % preprint build: authors shown, header reads 'Preprint'"),
    ("research licences; its videos are Internet-sourced footage",
     "research licences; their videos are Internet-sourced footage"),
    ("collected by the dataset authors, and we add no new personal data and release no RGB\n"
     "frames.",
     "collected by the dataset authors. We add no new personal data and do not redistribute\n"
     "either dataset. For reference, our figures show frames from DanceTrack validation videos,\n"
     f"and the project page (\\url{{{PAGE}}}) shows one full validation video and a short excerpt\n"
     "of another; all are credited to the dataset authors \\citep{sun2022dancetrack}."),
    (r"\usepackage[hidelinks]{hyperref}",
     r"\usepackage[hidelinks]{hyperref}" "\n"
     r"\hypersetup{pdftitle={The Generator Is the Tracker: Multi-Object Tracking by Painting"
     r" Persistent Identity Colours}, pdfauthor={Haiyu Yang, Miel Hostens}}"),
    ("All experiments are logged in a pinned experiment record (config hashes, seeds, costs, and\n"
     "gate verdicts per run) released with the code, ID-map codec, training and inference\n"
     "configurations, and evaluation scripts in the anonymised supplementary material\n"
     "submitted with this paper.",
     "All experiments were logged in a pinned experiment record (config hashes, seeds, costs,\n"
     "and gate verdicts per run). The code, ID-map codec, training and inference configurations,\n"
     f"evaluation scripts and run parameters are at \\url{{{REPO}}}; the LoRA checkpoints\n"
     f"are at \\url{{{HF}}}."),
    ("release code, checkpoints, and the full pinned experiment log.", "release code and checkpoints."),
    ("means fixed in a versioned, released log before the", "means fixed in a versioned log before the"),
    ("with the submission archive and per-sequence outputs included in the release.",
     "with the submission archive and per-sequence outputs included in the repository."),
]

tex = (SRC / "gentrack_iclr.tex").read_text(encoding="utf-8")
for old, new in EDITS:
    n = tex.count(old)
    assert n == 1, (n, old[:70])
    tex = tex.replace(old, new)
assert r"\PEND{" not in tex and "anonym" not in tex.lower(), "anonymity text left"

DST.mkdir(exist_ok=True)
(DST / "gentrack_arxiv.tex").write_text(tex, encoding="utf-8", newline="")
for f in STYLE:
    shutil.copy(SRC / f, DST / f)
sty = (SRC / "iclr2027_conference.sty").read_text(encoding="utf-8")
old = r"\lhead{Published as a conference paper at ICLR 2027}"
assert sty.count(old) == 1
(DST / "iclr2027_conference.sty").write_text(sty.replace(old, r"\lhead{Preprint}"), encoding="utf-8", newline="")
print("wrote", DST)
