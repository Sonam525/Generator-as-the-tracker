"""EXPLORATORY (post hoc, not pre-registered, 2026-11-10): person width vs the 32-px latent token, and a
one-frame (frame 25) breakdown of the generated masks per density-probe clip. Crowded-scene diagnosis only.

Usage: python -I paper/scripts/density_probe_exploratory.py <dir>
<dir> must hold, per clip c in CLIPS (order of the `clips` parameter in
experiments/jobs/paper_runs/reb_density_probe.json, sample i = position + 1):
  gt_<c>.mkv   <- <VOL>/ltx_ds/phase2_train/idmap/dancetrack<c>_window.mkv            (FFV1, lossless)
  gen_<c>.png  <- <VOL>/checkpoints/ltx_dt_density_probe/samples/step_000000_<i>_png/0024.png
Mask classes for each generated mask: match = IoU >= 0.5 with some GT mask; merge = covers >= 30% of two or
more GT masks; partial = >= 50% of it lies inside one GT mask; other = none of these.
Output: <dir>/exploratory_widths.json (copied to paper/data/ltx_dt_density_probe/).
"""
import json, sys
import numpy as np, av
import imageio.v2 as iio
sys.path.insert(0, "core")
from gentrack.color_codec import decode
S = sys.argv[1]
CLIPS = ["0096_000001", "0020_000295", "0083_000001", "0082_000001", "0096_000393", "0080_000638", "0086_000295",
         "0082_000442", "0001_000001", "0002_000001", "0006_000001", "0008_000001"]
out = []
for c in CLIPS:
    with av.open(f"{S}/dprobe/gt_{c}.mkv") as cont:
        frames = [fr.to_ndarray(format="rgb24") for fr in cont.decode(video=0)]
    widths = []
    for f in frames:
        for m in decode(f, K=48, min_area=150).values():
            xs = np.where(m.any(0))[0]
            widths.append(xs.max() - xs.min() + 1)
    g = np.asarray(iio.imread(f"{S}/dprobe/gen_{c}.png"))[:, :, :3]
    if g.shape[1] == 2 * g.shape[0]:
        g = g[:, g.shape[1] // 2:]
    gm, tm = decode(g, K=48, min_area=150), decode(frames[24], K=48, min_area=150)
    cls = dict(match=0, merge=0, partial=0, other=0)
    for gk in gm.values():
        st = [(np.logical_and(gk, t).sum() / max(np.logical_or(gk, t).sum(), 1), np.logical_and(gk, t).sum() / max(t.sum(), 1),
               np.logical_and(gk, t).sum() / max(gk.sum(), 1)) for t in tm.values()]
        if any(i >= 0.5 for i, _, _ in st): cls["match"] += 1
        elif sum(1 for _, ct, _ in st if ct >= 0.3) >= 2: cls["merge"] += 1
        elif any(it >= 0.5 for _, _, it in st): cls["partial"] += 1
        else: cls["other"] += 1
    r = dict(clip=c, gt_width_px_median=float(np.median(widths)), gt_width_tokens=round(float(np.median(widths)) / 32, 2),
             frac_people_under_one_token=round(float(np.mean(np.asarray(widths) < 32)), 2), f25_gt=len(tm), f25_generated=len(gm), **cls)
    out.append(r); print(r)
json.dump(out, open(f"{S}/dprobe/exploratory_widths.json", "w"), indent=1)
