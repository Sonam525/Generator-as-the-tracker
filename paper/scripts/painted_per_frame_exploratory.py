# EXPLORATORY (post hoc, 2026-11-10): painted regions per frame on validation, per factorial cell,
# from TrackEval Count fields (Dets / CLR_Frames) under the raw-colour link (one box per colour per frame).
import csv, io, json, sys
SRC = {"A chain/prefix-trained": ("paper/data/ltx_revision_A/trackeval_detailed.json", "ltx_arm_a_chain__native_raw_color"),
       "B indep/translation": ("paper/data/ltx_revision_B/trackeval_detailed.json", "ltx_arm_b_indep__native_raw_color"),
       "D indep/prefix-trained": ("paper/data/ltx_reb_armD/trackeval_detailed.json", "ltx_arm_d_indep_ckptB__native_raw_color"),
       "C chain/translation": ("paper/data/ltx_reb_armC/trackeval_detailed.json", "ltx_arm_c_chain_ckptA__native_raw_color")}
rows = {}
for cell, (path, key) in SRC.items():
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print("missing", cell); continue
    if key not in d:
        print("missing key", cell, list(d)); continue
    for r in csv.DictReader(io.StringIO(d[key])):
        if r["seq"] == "COMBINED":
            continue
        f = float(r["CLR_Frames"])
        rows.setdefault(r["seq"], {"gt": float(r["GT_Dets"]) / f})[cell] = float(r["Dets"]) / f
cells = [c for c in SRC if any(c in v for v in rows.values())]
print("video          GT/f  " + "  ".join(f"{c[:22]:>22}" for c in cells))
for v, r in sorted(rows.items(), key=lambda x: -x[1]["gt"]):
    print(f"{v}  {r['gt']:5.1f}  " + "  ".join(f"{r.get(c, float('nan')):22.1f}" for c in cells))
for band, lo, hi in [(">=14", 14, 99), ("<=9", 0, 9.5)]:
    sel = [r for r in rows.values() if lo <= r["gt"] < hi]
    print(band, len(sel), "videos; mean painted/f:", {c[:1]: round(sum(r[c] for r in sel if c in r) / max(1, sum(c in r for r in sel)), 1) for c in cells},
          "max per-video:", {c[:1]: round(max(r[c] for r in sel if c in r), 1) for c in cells})
