"""Numbers for the November 2026 revision (pre-registered 2026-11-09/10).

Reads merged run records built by make_revision_tables.py:
  paper/data/ltx_reb_factorial_merged  <- --out=ltx_reb_factorial ltx_reb_armC ltx_reb_armD ltx_revision_A ltx_revision_B ltx_reb_mech_A
  paper/data/ltx_reb_seeds_merged      <- --out=ltx_reb_seeds ltx_reb_seeds_all25 ltx_reb_seeds_assoc
and prints (and writes rebuttal_analysis.json next to the factorial record):
  1. the 2x2 factorial, every cell under both identity links and the two associators on its own boxes;
  2. factor effects as paired per-video bootstrap CIs (make_rebuttal_stats.boot, 10,000 resamples, seed 0);
  3. seeds 0-2 on all 25 videos: COMBINED mean and sd, re-acquisition, colour vs associators per seed;
  4. carry-over mechanism statistics per chain (pinned), plus the post-hoc exploratory occluder field.
Missing records are reported as missing, never filled in.
"""
import csv
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_rebuttal_stats import boot  # noqa: E402

FACT = "paper/data/ltx_reb_factorial_merged"
SEEDS = "paper/data/ltx_reb_seeds_merged"
M = ["HOTA", "DetA", "AssA", "IDF1"]
COLS = {"HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC", "IDF1": "IDF1"}
CELLS = {  # cell: (arm, checkpoint, generation, native link)
    "A": ("ltx_arm_a_chain", "prefix-trained", "chain", "raw_color"),
    "C": ("ltx_arm_c_chain_ckptA", "translation-only", "chain", "raw_color"),
    "D": ("ltx_arm_d_indep_ckptB", "prefix-trained", "independent", "v3many"),
    "B": ("ltx_arm_b_indep", "translation-only", "independent", "v3many"),
}
out = {}


def load(d, f):
    p = f"{d}/{f}"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def per_video(det, tag):
    if not det or tag not in det:
        return None
    rows = csv.DictReader(io.StringIO(det[tag]))
    return {r["seq"]: {m: 100 * float(r[COLS[m]]) for m in M} for r in rows if r["seq"] != "COMBINED"}


def paired(det, x, y):
    px, py = per_video(det, x), per_video(det, y)
    if px is None or py is None:
        return None
    vids = sorted(set(px) & set(py))
    r = {}
    for m in M:
        diff = [px[v][m] - py[v][m] for v in vids]
        mean, lo, hi = boot(diff)
        r[m] = dict(diff=round(mean, 2), ci95=[round(lo, 2), round(hi, 2)], favour_x=sum(d > 0 for d in diff), n=len(vids))
    return r


def fmt(r):
    return "  ".join(f"{m} {r[m]['diff']:+6.2f} [{r[m]['ci95'][0]:+6.2f},{r[m]['ci95'][1]:+6.2f}] {r[m]['favour_x']}/{r[m]['n']}" for m in M)


# ---------------------------------------------------------------- 1-2. factorial
te, det, rs = load(FACT, "trackeval_summary.json"), load(FACT, "trackeval_detailed.json"), load(FACT, "reacq_summary.json")
if te is None:
    print(f"missing {FACT}; run make_revision_tables.py first")
else:
    print("=== 1. Factorial, TrackEval COMBINED (25 val) ===")
    print(f"{'cell':4s} {'tracker':38s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s} {'IDF1':>6s} {'IDSW':>6s}  re-acq k/n")
    out["factorial"] = {}
    for cell, (arm, ck, gen, link) in CELLS.items():
        for tr in ["native_raw_color", "native_v3many", "kalman_iou", "bank_dino"]:
            tag = f"{arm}__{tr}"
            c = te.get(tag)
            if c is None:
                print(f"{cell:4s} {tag:38s} MISSING"); continue
            ra = (rs or {}).get("summary", {}).get(tag, {}).get("ALL", {})
            out["factorial"][tag] = dict(cell=cell, checkpoint=ck, generation=gen, native=tr == f"native_{link}",
                                         **{k: c.get(k) for k in M + ["IDSW", "DetRe", "DetPr", "AssRe", "AssPr"]},
                                         reacq=ra.get("reacq"), cond_n=ra.get("cond_n"), uncond=ra.get("uncond_rate"))
            star = "*" if tr == f"native_{link}" else " "
            print(f"{cell:4s} {tag+star:38s} {c['HOTA']:6.1f} {c['DetA']:6.1f} {c['AssA']:6.1f} {c['IDF1']:6.1f} {c['IDSW']:6.0f}  "
                  f"{ra.get('reacq')}/{ra.get('cond_n')} (unc {ra.get('uncond_rate')})")
    print("(* = the cell's native identity link)")

    def t(cell, tr):
        return f"{CELLS[cell][0]}__{tr}"
    nat = {c: f"native_{CELLS[c][3]}" for c in CELLS}
    comps = {
        # generation mode within a checkpoint
        "mode | prefix-trained ckpt: A chain - D indep (native links)": (t("A", nat["A"]), t("D", nat["D"])),
        "mode | prefix-trained ckpt: A - D (both raw colour)": (t("A", "native_raw_color"), t("D", "native_raw_color")),
        "mode | prefix-trained ckpt: A - D (both v3many)": (t("A", "native_v3many"), t("D", "native_v3many")),
        "mode | translation ckpt: C chain - B indep (native links)": (t("C", nat["C"]), t("B", nat["B"])),
        "mode | translation ckpt: C - B (both raw colour)": (t("C", "native_raw_color"), t("B", "native_raw_color")),
        "mode | translation ckpt: C - B (both v3many)": (t("C", "native_v3many"), t("B", "native_v3many")),
        # checkpoint within a generation mode
        "ckpt | chain: A prefix - C translation (raw colour)": (t("A", "native_raw_color"), t("C", "native_raw_color")),
        "ckpt | chain: A - C (v3many)": (t("A", "native_v3many"), t("C", "native_v3many")),
        "ckpt | indep: D prefix - B translation (v3many)": (t("D", "native_v3many"), t("B", "native_v3many")),
        "ckpt | indep: D - B (raw colour)": (t("D", "native_raw_color"), t("B", "native_raw_color")),
        # the published pair, for reference
        "published | A - B (native links)": (t("A", nat["A"]), t("B", nat["B"])),
    }
    for c in CELLS:  # association at fixed detection inside each cell
        comps[f"same boxes | {c}: colours - Kalman/IoU"] = (t(c, nat[c]), t(c, "kalman_iou"))
        comps[f"same boxes | {c}: colours - DINOv3 bank"] = (t(c, nat[c]), t(c, "bank_dino"))
    print("\n=== 2. Paired per-video bootstrap, X - Y (mean diff [95% CI] videos favouring X) ===")
    out["effects"] = {}
    for name, (x, y) in comps.items():
        r = paired(det, x, y)
        out["effects"][name] = r
        print(f"{name:58s} " + (fmt(r) if r else "MISSING"))

# ---------------------------------------------------------------- 3. seeds
SEED_TAGS = {0: "ltx_arm_a_chain", 1: "ltx_seed1_chain", 2: "ltx_seed2_chain"}
ste, sdet, srs = load(SEEDS, "trackeval_summary.json"), load(SEEDS, "trackeval_detailed.json"), load(SEEDS, "reacq_summary.json")
if ste is None:
    print(f"\nmissing {SEEDS}")
else:
    print("\n=== 3. Seeds, all 25 val videos, colour (raw colour link) ===")
    rows = {}
    for s, arm in SEED_TAGS.items():
        tag = f"{arm}__native_raw_color"
        c = ste.get(tag) or (te or {}).get(tag)
        ra = (srs or {}).get("summary", {}).get(tag, {}).get("ALL", {})
        if c is None:
            print(f"seed {s}: MISSING"); continue
        rows[s] = {k: c[k] for k in M}
        rows[s].update(reacq=ra.get("reacq"), cond_n=ra.get("cond_n"), carry=ra.get("carry_over_frac_of_reacq"))
        print(f"seed {s}: " + " ".join(f"{k} {c[k]:.1f}" for k in M) + f"  re-acq {ra.get('reacq')}/{ra.get('cond_n')} carry {ra.get('carry_over_frac_of_reacq')}")
    if len(rows) == 3:
        agg = {k: (round(float(np.mean([rows[s][k] for s in rows])), 2), round(float(np.std([rows[s][k] for s in rows], ddof=1)), 2)) for k in M}
        print("mean (sd, ddof=1): " + "  ".join(f"{k} {m:.1f} ({sd:.1f})" for k, (m, sd) in agg.items()))
        out["seeds"] = dict(per_seed=rows, mean_sd=agg)
    out["seeds_same_boxes"] = {}
    for s, arm in SEED_TAGS.items():
        D_ = sdet if s else det
        for other in ["kalman_iou", "bank_dino"]:
            r = paired(D_, f"{arm}__native_raw_color", f"{arm}__{other}")
            out["seeds_same_boxes"][f"seed{s}: colours - {other}"] = r
            print(f"seed {s} colours - {other:10s} " + (fmt(r) if r else "MISSING"))

# ---------------------------------------------------------------- 4. mechanism
# Arm A is in two records with identical decodes (ltx_reb_mech_A, and the seeds run that adds the exploratory
# field). Events carry no unique key (one person can have two gaps of equal length), so de-duplicate by record:
# a tag present in the seeds record is taken from there only.
fmech, smech = load(FACT, "mechanism_events.json") or [], load(SEEDS, "mechanism_events.json") or []
stags = {m["tag"] for m in smech}
uniq = [m for m in fmech if m["tag"] not in stags] + smech
print("\n=== 4. Carry-over mechanism, raw-colour link (medians) ===")
out["mechanism"] = {}


def med(sel, k):
    v = [m[k] for m in sel if m.get(k) is not None]
    return (round(float(np.median(v)), 2), len(v)) if v else (None, 0)


for tag in sorted({m["tag"] for m in uniq}):
    for lab, flag in [("re-acquired", True), ("not re-acquired", False)]:
        sel = [m for m in uniq if m["tag"] == tag and m["reacq"] == flag]
        if not sel:
            continue
        pr = [m["prefix_frames_with_colour"] for m in sel if m["prefix_frames_with_colour"]]
        row = dict(n=len(sel), present_ge1=sum(m["frac_present"] > 0 for m in sel),
                   **{k: med(sel, k)[0] for k in ["frac_present", "longest_run", "n_runs", "n_gap", "rel_area_mean",
                                                  "iou_interp_mean", "frac_on_other", "ncomp_mean"]},
                   spans_window_start=len(pr), every_prefix=sum(all(x > 0 for x in p) for p in pr),
                   every_prefix_all9=sum(all(x == 9 for x in p) for p in pr), no_prefix=sum(all(x == 0 for x in p) for p in pr))
        x = [m["frac_other_keeps_own"] for m in sel if m.get("frac_other_keeps_own") is not None]
        row["EXPLORATORY_other_keeps_own"] = dict(median=round(float(np.median(x)), 2), mean=round(float(np.mean(x)), 2), n=len(x)) if x else None
        out["mechanism"][f"{tag} | {lab}"] = row
        print(f"{tag} | {lab}: " + ", ".join(f"{k}={v}" for k, v in row.items()))
    bx = [m for m in uniq if m["tag"] == tag and m["reacq"] and m["n_gap"] >= 49]
    if bx:
        print(f"  beyond-context re-acquisitions ({len(bx)}): " + "; ".join(
            f"{m['vid'][-4:]}/id{m['tid']} gap {m['n_gap']} pres {m['frac_present']:.2f} pref {m['prefix_frames_with_colour']} "
            f"iouI {m['iou_interp_mean']} area {m['rel_area_mean']} keeps_own {m.get('frac_other_keeps_own')}" for m in bx))

if os.path.isdir(FACT):
    json.dump(out, open(f"{FACT}/rebuttal_analysis.json", "w"), indent=1)
    print(f"\nwrote {FACT}/rebuttal_analysis.json")
