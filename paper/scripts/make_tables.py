"""Build the revision's numeric tables from logged run records (paper/data/*).

Emits LaTeX snippets into paper/tables/ and prints the headline numbers.
Inputs (downloaded from the volume, results/ltx_*):
  paper/data/ltx_arm_{a_chain,b_indep}_eval_trackeval_stdout.txt   TrackEval summary tables
  paper/data/ltx_arm_a_chain_eval_metrics_raw_color.json           per-video pred/gt tracks (Arm A)
  paper/data/ltx_arm_b_indep_eval_metrics_v3many.json              per-video pred/gt tracks (Arm B)
  paper/assets/reacq_events.json                                    383 gap events x 2 arms
"""
import io
import json
import math
import os
from collections import Counter

D = "paper/data"
OUT = "paper/tables"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- TrackEval parsing
def parse_trackeval(path):
    """Return {vid: {metric: value}} for the HOTA, CLEAR and Identity blocks (COMBINED included)."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    per = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(("HOTA:", "CLEAR:", "Identity:")):
            hdr = ln.split()[1:]           # first token is 'HOTA: <tracker>-pedestrianHOTA' -> split gives it
            # header row is on the same line after the tracker name; find the metric names
            toks = ln.replace(":", " ").split()
            names = [t for t in toks if t not in ("HOTA:", "CLEAR:", "Identity:")]
            # metric names start after the tracker token (contains '-pedestrian')
            k = next(j for j, t in enumerate(toks) if "pedestrian" in t)
            names = toks[k + 1:]
            # the first name is glued: e.g. 'ltx_arm_a_chain_raw_color-pedestrianHOTA' -> strip prefix
            glued = toks[k]
            first = glued.split("pedestrian", 1)[1]
            names = [first] + names if first else names
            i += 1
            while i < len(lines) and lines[i].strip():
                row = lines[i].split()
                vid, vals = row[0], row[1:]
                per.setdefault(vid, {})
                for n, v in zip(names, vals):
                    try:
                        per[vid][n] = float(v)
                    except ValueError:
                        pass
                i += 1
        i += 1
    return per


A = parse_trackeval(f"{D}/ltx_arm_a_chain_eval_trackeval_stdout.txt")
B = parse_trackeval(f"{D}/ltx_arm_b_indep_eval_trackeval_stdout.txt")
metaA = json.load(open(f"{D}/ltx_arm_a_chain_eval_metrics_raw_color.json"))["per_video"]
metaB = json.load(open(f"{D}/ltx_arm_b_indep_eval_metrics_v3many.json"))["per_video"]
VIDS = sorted(v for v in A if v != "COMBINED")
assert len(VIDS) == 25, VIDS

SUB = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "IDF1", "MOTA"]
print("COMBINED Arm A:", {k: A["COMBINED"][k] for k in SUB})
print("COMBINED Arm B:", {k: B["COMBINED"][k] for k in SUB})

# ---------------------------------------------------------------- per-video appendix table
rows = []
for v in VIDS:
    a, b = A[v], B[v]
    rows.append(
        f"{v[-4:]} & {metaA[v]['gt_tracks']} & {metaA[v]['frames']} & "
        f"{a['HOTA']:.1f} & {a['DetA']:.1f} & {a['AssA']:.1f} & {a['IDF1']:.1f} & {metaA[v]['pred_tracks']} & "
        f"{b['HOTA']:.1f} & {b['DetA']:.1f} & {b['AssA']:.1f} & {b['IDF1']:.1f} & {metaB[v]['pred_tracks']} \\\\")
c = A["COMBINED"]; cb = B["COMBINED"]
rows.append("\\midrule")
rows.append(f"all & -- & -- & {c['HOTA']:.1f} & {c['DetA']:.1f} & {c['AssA']:.1f} & {c['IDF1']:.1f} & "
            f"{sum(metaA[v]['pred_tracks'] for v in VIDS)} & {cb['HOTA']:.1f} & {cb['DetA']:.1f} & "
            f"{cb['AssA']:.1f} & {cb['IDF1']:.1f} & {sum(metaB[v]['pred_tracks'] for v in VIDS)} \\\\")
rows.append("\\bottomrule")
io.open(f"{OUT}/per_video.tex", "w", newline="\n").write("\n".join(rows) + "\n")

# ---------------------------------------------------------------- sub-metric decomposition table
sub_rows = []
for name, t in (("A: generation-carried (chained)", A["COMBINED"]), ("B: independent + post-hoc association", B["COMBINED"])):
    sub_rows.append(f"{name} & " + " & ".join(f"{t[k]:.1f}" for k in SUB) + " \\\\")
io.open(f"{OUT}/submetrics.tex", "w", newline="\n").write("\n".join(sub_rows) + "\n")

# ---------------------------------------------------------------- density correlation (Spearman)
def spearman(x, y):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


people = [metaA[v]["gt_tracks"] for v in VIDS]
print("Spearman(people, AssA_A) =", round(spearman(people, [A[v]["AssA"] for v in VIDS]), 3))
print("Spearman(people, DetA_A) =", round(spearman(people, [A[v]["DetA"] for v in VIDS]), 3))
print("Spearman(people, HOTA_A) =", round(spearman(people, [A[v]["HOTA"] for v in VIDS]), 3))
sparse = [v for v in VIDS if metaA[v]["gt_tracks"] <= 9]
dense = [v for v in VIDS if metaA[v]["gt_tracks"] >= 14]
print(f"sparse (<=9 people, n={len(sparse)}): mean AssA {sum(A[v]['AssA'] for v in sparse)/len(sparse):.1f}, "
      f"DetA {sum(A[v]['DetA'] for v in sparse)/len(sparse):.1f}; dense (>=14, n={len(dense)}): mean AssA "
      f"{sum(A[v]['AssA'] for v in dense)/len(dense):.1f}, DetA {sum(A[v]['DetA'] for v in dense)/len(dense):.1f}")
ratioA = sum(metaA[v]["pred_tracks"] for v in VIDS) / sum(metaA[v]["gt_tracks"] for v in VIDS)
ratioB = sum(metaB[v]["pred_tracks"] for v in VIDS) / sum(metaB[v]["gt_tracks"] for v in VIDS)
print(f"predicted ids per GT id: A {ratioA:.2f}, B {ratioB:.2f}; total GT tracks {sum(metaA[v]['gt_tracks'] for v in VIDS)}")

# ---------------------------------------------------------------- re-acquisition statistics
# Source: the revision run (results/ltx_revision_{A,B}), whose per-frame boxes are exactly the
# union-merged boxes TrackEval scores; tags = tracker names. The 2026-07-11 study scored arm B on
# pre-union boxes (255 conditional events); the corrected protocol gives 154.
ev = json.load(open("paper/data/ltx_revision_merged/reacq_events.json"))
BINS = [("9--24", 9, 24), ("25--48", 25, 48), ("49--150", 49, 150), ("$>$150", 151, 10 ** 9)]
TAGS = [("A: chain (ours)", "ltx_arm_a_chain__native_raw_color"),
        ("\quad A boxes, Kalman/IoU", "ltx_arm_a_chain__kalman_iou"),
        ("\quad A boxes, appearance$+$motion bank", "ltx_arm_a_chain__bank_dino"),
        ("B: independent $+$ assoc.", "ltx_arm_b_indep__native_v3many"),
        ("\quad B raw colour (no linking)", "ltx_arm_b_indep__native_raw_color")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def stats(tag):
    rows = [e for e in ev if e["tag"] == tag]
    out = {}
    for lbl, lo, hi in BINS + [("all", 9, 10 ** 9)]:
        sel = [e for e in rows if lo <= e["gap"] <= hi]
        cond = [e for e in sel if e["tracked_before"] and e["redetect"]]
        k = sum(e["reacq"] for e in cond)
        ch = [e["chance"] for e in cond if e["chance"]]
        carry = [e["pre_id_present_during_gap"] for e in cond if e["reacq"] and e["pre_id_present_during_gap"] is not None]
        out[lbl] = dict(n_all=len(sel), n_cond=len(cond), k=k,
                        cond_rate=(k / len(cond)) if cond else float("nan"),
                        uncond_rate=(k / len(sel)) if sel else float("nan"),
                        ci=wilson(k, len(cond)),
                        chance=(sum(ch) / len(ch)) if ch else float("nan"),
                        carry=(sum(carry) / len(carry)) if carry else float("nan"))
    return out


S = {tag: stats(tag) for _, tag in TAGS}
for name, tag in TAGS:
    print(f"--- {name} [{tag}]")
    for lbl, _, _ in BINS + [("all", 0, 0)]:
        a = S[tag][lbl]
        print(f"  {lbl:8s} events {a['n_all']:3d} | {a['k']:2d}/{a['n_cond']:3d} = {a['cond_rate']:.3f} "
              f"[{a['ci'][0]:.2f},{a['ci'][1]:.2f}] uncond {a['uncond_rate']:.3f} chance {a['chance']:.3f} carry {a['carry']:.2f}")


def paired(tag1, tag2):
    # Pair by position: (vid, tid, gap) is not unique (17 of 383 events repeat a key), and keying
    # on it silently dropped events (correction logged 2026-11-11).
    e1, e2 = [e for e in ev if e["tag"] == tag1], [e for e in ev if e["tag"] == tag2]
    assert [(e["vid"], e["tid"], e["gap"]) for e in e1] == [(e["vid"], e["tid"], e["gap"]) for e in e2]
    byk = [{tag1: a, tag2: b} for a, b in zip(e1, e2)]
    shared = [d for d in byk if all(d[t]["tracked_before"] and d[t]["redetect"] for t in (tag1, tag2))]
    a_only = sum(1 for d in shared if d[tag1]["reacq"] and not d[tag2]["reacq"])
    b_only = sum(1 for d in shared if d[tag2]["reacq"] and not d[tag1]["reacq"])
    nd = a_only + b_only
    p_two = min(1.0, 2 * sum(math.comb(nd, i) for i in range(0, min(a_only, b_only) + 1)) / 2 ** nd) if nd else 1.0
    return len(shared), a_only, b_only, p_two


PAIRS = {}
for other in ["ltx_arm_b_indep__native_v3many", "ltx_arm_a_chain__kalman_iou", "ltx_arm_a_chain__bank_dino"]:
    n, a1, b1, p = paired("ltx_arm_a_chain__native_raw_color", other)
    PAIRS[other] = dict(shared=n, a_only=a1, other_only=b1, p=p)
    print(f"paired A-native vs {other}: shared {n}, A-only {a1}, other-only {b1}, exact McNemar p = {p:.3f}")

kinds = Counter(e["kind"] for e in ev if e["tag"] == "ltx_arm_a_chain__native_raw_color")
print("event kinds:", dict(kinds))

# LaTeX table: k/n per system per bin, plus rate/CI and unconditional for the two arms
tex = []
tex.append("Gap length (frames) & " + " & ".join(l for l, _, _ in BINS) + " & all \\\\")
tex.append("\midrule")
SA = S["ltx_arm_a_chain__native_raw_color"]
tex.append("GT gap events & " + " & ".join(str(SA[l]["n_all"]) for l, _, _ in BINS) + f" & {SA['all']['n_all']} \\\\")
for name, tag in TAGS:
    T = S[tag]
    tex.append(f"{name}, $k/n$ & " + " & ".join(f"{T[l]['k']}/{T[l]['n_cond']}" for l, _, _ in BINS) + f" & {T['all']['k']}/{T['all']['n_cond']} \\\\")
    if tag in ("ltx_arm_a_chain__native_raw_color", "ltx_arm_b_indep__native_v3many"):
        tex.append("\quad rate (95\% CI) & " +
                   " & ".join((f"{T[l]['cond_rate']:.2f} ({T[l]['ci'][0]:.2f}--{T[l]['ci'][1]:.2f})" if T[l]['n_cond'] >= 5
                               else f"{T[l]['k']}/{T[l]['n_cond']}") for l, _, _ in BINS) +
                   f" & {T['all']['cond_rate']:.2f} ({T['all']['ci'][0]:.2f}--{T['all']['ci'][1]:.2f}) \\\\")
        tex.append("\quad unconditional & " + " & ".join(f"{T[l]['uncond_rate']:.2f}" for l, _, _ in BINS) + f" & {T['all']['uncond_rate']:.2f} \\\\")
tex.append("\\bottomrule")
io.open(f"{OUT}/reacq.tex", "w", newline="\n").write("\n".join(tex) + "\n")
json.dump(dict(stats={t: S[t] for _, t in TAGS}, pairs=PAIRS,
               spearman_people_assA=spearman(people, [A[v]["AssA"] for v in VIDS]),
               spearman_people_detA=spearman(people, [A[v]["DetA"] for v in VIDS]),
               pred_per_gt_A=ratioA, pred_per_gt_B=ratioB),
          open(f"{OUT}/summary_numbers.json", "w"), indent=1, default=str)
print("wrote", OUT)
