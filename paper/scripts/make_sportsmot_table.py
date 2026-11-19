"""Build the SportsMOT generalisation table from the logged run records.

Inputs (created by `python paper/scripts/make_revision_tables.py <run>`):
  paper/data/ltx_sm_zeroshot_eval_merged/   zero-shot transfer, DanceTrack checkpoint, VAL8
  paper/data/ltx_sm_val_eval_merged/        SportsMOT fine-tune (phase A + B), all 45 val videos

Emits paper/tables/sportsmot.tex and prints every number that goes into the prose.
Runs with whichever inputs exist, so it is usable before the fine-tuned arm lands.

Reporting rule, pinned 2026-09-08 BEFORE the fine-tuned numbers existed: the headline is the
TrackEval COMBINED figure over all 45 val videos, and because the 8-video in-loop subset also
chose the checkpoint, a second column gives the unweighted mean over the 37 videos that took no
part in selection. A per-video mean is not a COMBINED score (TrackEval aggregates raw counts,
and AssA is not reconstructible from per-video values), so it is labelled as a mean and the two
columns are never mixed.
"""
import csv
import io
import json
import os

D = "paper/data"
OUT = "paper/tables"
os.makedirs(OUT, exist_ok=True)

# The 8 videos used for in-loop validation and checkpoint selection (pre-registered).
VAL8 = ["v_00HRwkvvjtQ_c001", "v_00HRwkvvjtQ_c003", "v_00HRwkvvjtQ_c005", "v_2QhNRucNC7E_c017",
        "v_G-vNjfx1GGc_c004", "v_G-vNjfx1GGc_c008", "v_0kUtTtmLaJA_c004", "v_0kUtTtmLaJA_c005"]

TRACKERS = [("native_raw_color", "generated colour"),
            ("kalman_iou", "Kalman/IoU on the same boxes"),
            ("bank_dino", "DINOv3 bank on the same boxes")]
REGIMES = [("ltx_sm_zeroshot_eval_merged", "ltx_sm_zeroshot_chain", "Zero-shot (DanceTrack checkpoint)"),
           ("ltx_sm_val_eval_merged", "ltx_sm_arm_a_chain", "Fine-tuned on SportsMOT train")]
KEYS = [("HOTA___AUC", "HOTA"), ("DetA___AUC", "DetA"), ("AssA___AUC", "AssA"), ("IDF1", "IDF1")]


def per_video(detailed, tag):
    """{seq: {metric: value*100}} from the TrackEval detailed CSV held in the run record."""
    if tag not in detailed:
        return {}
    rows = list(csv.DictReader(io.StringIO(detailed[tag])))
    return {r["seq"]: {name: 100 * float(r[col]) for col, name in KEYS if r.get(col)} for r in rows}


def load(run_dir):
    def read(f):
        p = f"{D}/{run_dir}/{f}"
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    return read("trackeval_summary.json"), read("trackeval_detailed.json"), read("reacq_summary.json")


rows, notes = [], []
for run_dir, arm, label in REGIMES:
    summary, detailed, reacq = load(run_dir)
    if not summary:
        notes.append(f"MISSING {run_dir} -- run: python paper/scripts/make_revision_tables.py {run_dir[:-7]}")
        continue
    pv_all = {t: per_video(detailed or {}, f"{arm}__{t}") for t, _ in TRACKERS}
    seqs = [s for s in next(iter(pv_all.values())) if s != "COMBINED"]
    held = sorted(set(seqs) - set(VAL8))
    print(f"\n=== {label}: {len(seqs)} videos scored, {len(held)} outside the selection subset")
    for tag, tname in TRACKERS:
        key = f"{arm}__{tag}"
        if key not in summary:
            continue
        comb = {name: summary[key][name] for _, name in KEYS if name in summary[key]}
        pv = pv_all[tag]
        mean = {name: sum(pv[s][name] for s in held) / len(held) for _, name in KEYS} if held else {}
        ra = ""
        if reacq and key in reacq.get("summary", {}):
            a = reacq["summary"][key]["ALL"]
            ra = f"{a['reacq']}/{a['cond_n']}"
        rows.append((label, tname, comb, mean, ra, len(seqs), len(held)))
        print(f"  {tname:32s} COMBINED " + " ".join(f"{n} {comb.get(n, float('nan')):5.1f}" for _, n in KEYS) +
              (f" | held-out mean " + " ".join(f"{n} {mean[n]:5.1f}" for _, n in KEYS) if mean else "") +
              (f" | re-acq {ra}" if ra else ""))

# No \toprule here: the document supplies it before the \input, matching tables/reacq.tex. A
# \noalign command at the very start of an \input file inside tabular is a "Misplaced \noalign".
lines = [r"& \multicolumn{4}{c}{all val videos (TrackEval combined)} & \multicolumn{4}{c}{held-out mean} & \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"association & HOTA & DetA & AssA & IDF1 & HOTA & DetA & AssA & IDF1 & re-acq \\"]
last = None
for label, tname, comb, mean, ra, n_all, n_held in rows:
    if label != last:
        lines.append(r"\midrule")
        scope = f"{n_all} videos; {n_held} held out" if n_held else f"{n_all} selection videos"
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{label}}} ({scope})}} \\")
        last = label
    cells = [f"{comb.get(n, float('nan')):.1f}" for _, n in KEYS] + \
            ([f"{mean[n]:.1f}" for _, n in KEYS] if mean else ["--"] * 4)
    lines.append(f"{tname} & " + " & ".join(cells) + f" & {ra or '--'} " + r"\\")
lines.append(r"\bottomrule")

path = f"{OUT}/sportsmot.tex"
with io.open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote", path)
for n in notes:
    print(n)
