"""Per-video seed table for the revision appendix (pre-registered 2026-11-09/10).

Reads the merged records built by make_revision_tables.py (factorial: seed 0 = arm A with its
Kalman/IoU and DINOv3-bank rows; seeds: arm A and seeds 1-2) and writes
paper/tables/seeds_per_video.tex: per video, HOTA and AssA of the colours for seeds
0/1/2, and AssA of Kalman/IoU and of the bank on each seed's own boxes. Missing records -> "--".
"""
import csv
import io
import json
import os

FACT = "paper/data/ltx_reb_factorial_merged/trackeval_detailed.json"
SEEDS = "paper/data/ltx_reb_seeds_merged/trackeval_detailed.json"
OUT = "paper/tables/seeds_per_video.tex"
ARMS = ["ltx_arm_a_chain", "ltx_seed1_chain", "ltx_seed2_chain"]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


rec = {**load(FACT), **load(SEEDS)}  # seeds record wins for arm A colours (same run re-scored)
rows = {}
for tag, s in rec.items():
    for r in csv.DictReader(io.StringIO(s)):
        rows.setdefault(r["seq"], {})[tag] = r


def cell(seq, arm, tr, col):
    r = rows.get(seq, {}).get(f"{arm}__{tr}")
    return f"{100 * float(r[col]):.1f}" if r else "--"


seqs = sorted(s for s in rows if s != "COMBINED") + ["COMBINED"]
lines = []
for seq in seqs:
    g = rows[seq].get("ltx_arm_a_chain__native_raw_color")
    people = g["GT_IDs"] if g and "GT_IDs" in g else "--"
    vals = [cell(seq, a, "native_raw_color", "HOTA___AUC") for a in ARMS]
    for tr in ["native_raw_color", "kalman_iou", "bank_dino"]:
        vals += [cell(seq, a, tr, "AssA___AUC") for a in ARMS]
    name = "combined" if seq == "COMBINED" else seq.replace("dancetrack", "")
    if seq == "COMBINED":
        lines.append("\\midrule")
    lines.append(" & ".join(f"\\rev{{{c}}}" for c in [name, str(people).split(".")[0]] + vals) + " \\\\")  # blue = revision
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines.append("\\bottomrule")  # \input followed by \bottomrule in the .tex fails (misplaced \noalign)
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"wrote {OUT}")
