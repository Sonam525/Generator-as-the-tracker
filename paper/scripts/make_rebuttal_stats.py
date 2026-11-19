"""Paired per-video bootstrap CIs for the revision (pre-registered 2026-11-09).

For each comparison (tracker X vs tracker Y scored on the same videos), resample videos with
replacement 10,000 times (numpy seed 0) and report the mean per-video difference with a 95%
percentile interval, plus how many videos favour X. Per-video values come from the TrackEval
detailed CSVs held in the merged run records (HOTA___AUC, AssA___AUC, IDF1).

Usage: python paper/scripts/make_rebuttal_stats.py <merged_dir> <tagX>:<tagY> [<tagX>:<tagY> ...]
"""
import csv
import io
import json
import sys

import numpy as np

KEYS = [("HOTA___AUC", "HOTA"), ("DetA___AUC", "DetA"), ("AssA___AUC", "AssA"), ("IDF1", "IDF1")]


def per_video(detailed, tag):
    rows = csv.DictReader(io.StringIO(detailed[tag]))
    return {r["seq"]: {n: 100 * float(r[c]) for c, n in KEYS} for r in rows if r["seq"] != "COMBINED"}


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    D = sys.argv[1]
    det = json.load(open(f"{D}/trackeval_detailed.json", encoding="utf-8"))
    out = {}
    for pair in sys.argv[2:]:
        x, y = pair.split(":")
        px, py = per_video(det, x), per_video(det, y)
        vids = sorted(set(px) & set(py))
        out[pair] = {}
        print(f"\n{x}  vs  {y}   ({len(vids)} videos)")
        for _, n in KEYS:
            diff = [px[v][n] - py[v][n] for v in vids]
            m, lo, hi = boot(diff)
            wins = sum(1 for d in diff if d > 0)
            out[pair][n] = dict(mean_diff=round(m, 2), ci95=[round(lo, 2), round(hi, 2)], videos_favouring_x=wins, n=len(vids))
            print(f"  {n:5s} diff {m:+6.2f}  95% CI [{lo:+6.2f}, {hi:+6.2f}]  favours X on {wins}/{len(vids)}")
    json.dump(out, open(f"{D}/bootstrap_ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()
