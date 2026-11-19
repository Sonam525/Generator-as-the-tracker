"""Pull results/<run> from the volume, print the numbers behind the revision placeholders, and
build fig_chain_decay.pdf. Usage: python paper/scripts/make_revision_tables.py [--out=NAME] [run ...]
Several runs merge into paper/data/<NAME or first run>_merged.
"""
import json
import os
import subprocess
import sys

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
RUNS = [a for a in sys.argv[1:] if not a.startswith("--out=")] or ["ltx_revision_A", "ltx_revision_B"]
VOL = "dbfs:/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker/results"
FILES = ["trackeval_summary.json", "reacq_summary.json", "window_curve.json", "union_merges.json", "trackeval_detailed.json", "reacq_events.json", "mechanism_events.json"]
merged = {f: {} for f in FILES}
for RUN in RUNS:
    D = f"paper/data/{RUN}"
    os.makedirs(D, exist_ok=True)
    for f in FILES:
        r = subprocess.run(["databricks", "fs", "cp", f"{VOL}/{RUN}/{f}", f"{D}/{f}", "--overwrite"], capture_output=True, text=True)
        if r.returncode:
            print("download failed:", RUN, f, r.stderr[-200:]); continue
        obj = json.load(open(f"{D}/{f}"))
        if f == "reacq_summary.json":
            merged[f].setdefault("summary", {}).update(obj["summary"]); merged[f]["params"] = obj["params"]
        elif f == "window_curve.json":
            if obj.get("aggregate"):
                merged[f] = obj
        elif f in ("reacq_events.json", "mechanism_events.json"):
            merged[f] = (merged[f] or []) + obj if isinstance(merged[f], list) else obj
        else:
            merged[f].update(obj)
DEFAULT = RUNS == ["ltx_revision_A", "ltx_revision_B"]
D = "paper/data/ltx_revision_merged" if DEFAULT else f"paper/data/{OUT or RUNS[0]}_merged"
os.makedirs(D, exist_ok=True)
for f in FILES:
    json.dump(merged[f], open(f"{D}/{f}", "w"))

te = json.load(open(f"{D}/trackeval_summary.json"))
KEYS = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "IDF1", "MOTA", "IDSW"]
print("\n=== TrackEval COMBINED (25 val) ===")
print(f"{'tracker':42s}" + "".join(f"{k:>7s}" for k in KEYS))
for t, c in te.items():
    print(f"{t:42s}" + "".join(f"{c.get(k, float('nan')):7.1f}" for k in KEYS))

rs = json.load(open(f"{D}/reacq_summary.json"))["summary"]
BINS = ["short9-24", "subwin25-48", "xwin49-150", "long>150", "ALL"]
print("\n=== Re-acquisition k/n (conditional), unconditional, chance, carry-over ===")
for t, s in rs.items():
    cells = []
    for b in BINS:
        d = s[b]
        cells.append(f"{d['reacq']}/{d['cond_n']} ({d['cond_rate']}) u{d['uncond_rate']}")
    print(f"{t:42s} " + " | ".join(cells))
    print(f"{'':42s} chance " + " | ".join(str(s[b]['mean_chance']) for b in BINS) +
          "   carry-over(of reacq) " + " | ".join(str(s[b]['carry_over_frac_of_reacq']) for b in BINS))

um = json.load(open(f"{D}/union_merges.json"))
agg = {}
for k, v in um.items():
    arm = k.split("/")[0]
    a = agg.setdefault(arm, dict(multi=0, boxes=0, frames=0))
    a["multi"] += v["multi_component_boxes"]; a["boxes"] += v["boxes"]; a["frames"] += v["frames"]
print("\n=== Same-frame multi-component colours (silent union merges) ===")
for arm, a in agg.items():
    print(f"{arm}: {a['multi']} of {a['boxes']} boxes ({100*a['multi']/max(1,a['boxes']):.1f}%) span >=2 components")

wc = json.load(open(f"{D}/window_curve.json"))["aggregate"]
if isinstance(wc, dict):
    wc = wc.get("ltx_arm_a_chain") or next(iter(wc.values()))
print("\n=== Per-window IDF1 (arm A) ===")
print(" ".join(f"w{c['wi']}:{c['idf1_mean']:.2f}(n={c['n']})" for c in wc))

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUT, ACC = "#1F2937", "#6B7280", "#2563EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUT, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT, "pdf.fonttype": 42})
wi = [c["wi"] + 1 for c in wc]
fig, ax1 = plt.subplots(figsize=(6.2, 2.4))
ax1.plot(wi, [c["idf1_mean"] for c in wc], color=ACC, lw=1.4, marker="o", ms=2.5, label="IDF1 (owned frames)")
ax1.set_xlabel("window index along the chain"); ax1.set_ylabel("IDF1", color=ACC)
ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.plot(wi, [c["ids_mean"] for c in wc], color=MUT, lw=1.2, ls="--", label="distinct ids in window")
ax2.plot(wi, [c["gt_ids_mean"] for c in wc], color=INK, lw=1.0, ls=":", label="GT people in window")
ax2.set_ylabel("ids per window", color=MUT)
for s in ("top",):
    ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
ax1.grid(True, color="#E5E7EB", lw=0.5)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
ax2.set_ylim(5, 19)
fig.tight_layout()
FIG = "paper/figures/fig_chain_decay.pdf" if DEFAULT else f"paper/figures/fig_chain_decay_{OUT or RUNS[0]}.pdf"
fig.savefig(FIG, dpi=300)
print("\nwrote", FIG)
