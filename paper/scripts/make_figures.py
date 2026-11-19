"""Build all paper figures (PDF, print-quality) from existing artifacts.
Style: single accent hue for 'ours' + neutral inks; identity via direct labels and
marker shape (grayscale/CVD-safe); ID-map panel colours are the data itself.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.path import Path
import numpy as np
from PIL import Image

A = "paper/assets"
F = "paper/figures"
PNG = os.environ.get("GENTRACK_FIG_PNG")  # optional dir: also write 200-dpi PNG check renders
os.makedirs(F, exist_ok=True)
INK, MUT, GRID, ACC = "#1F2937", "#6B7280", "#D1D5DB", "#2563EB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUT, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT,
    "pdf.fonttype": 42,
})


def sq(path, size=512):
    """Load image, squeeze to square (the model's input convention)."""
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))


def save(fig, name, **kw):
    """PDF into F; plus a PNG check render when GENTRACK_FIG_PNG is set."""
    fig.savefig(f"{F}/{name}.pdf", **kw)
    if PNG:
        fig.savefig(f"{PNG}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"{name}.pdf")


# ---------------------------------------------------------------- fig 2: chain strip
def fig_chain():
    frames = [1, 175, 350, 525, 700]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.15))
    for j, t in enumerate(frames):
        axes[0, j].imshow(sq(f"{A}/rgb_0065_f{t:04d}.jpg"))
        axes[1, j].imshow(sq(f"{A}/id_0065_f{t:04d}.png"))
        axes[0, j].set_title(f"frame {t}", fontsize=9, color=INK, pad=3)
        for i in (0, 1):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("input RGB", fontsize=9.5, color=INK)
    axes[1, 0].set_ylabel("generated ID-map", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.94, bottom=0.01,
                        wspace=0.03, hspace=0.04)
    fig.savefig(f"{F}/fig_chain.pdf", dpi=300)
    plt.close(fig)
    print("fig_chain.pdf")


# ---------------------------------------------------------------- fig 3: re-acquisition
def fig_reacq():
    VID, TID, PRE, MID, POST = "0019", 6, 488, 504, 521
    W0, H0 = Image.open(f"{A}/rgb_{VID}_pre.jpg").size
    cent = {}
    for ln in open(f"{A}/gt_{VID}.txt"):
        p = ln.split(",")
        f, tid = int(p[0]), int(p[1])
        if tid == TID and f in (PRE, POST):
            x, y, w, h = map(float, p[2:6])
            cent[f] = ((x + w / 2) * 512 / W0, (y + h / 2) * 512 / H0)
    mid = ((cent[PRE][0] + cent[POST][0]) / 2, (cent[PRE][1] + cent[POST][1]) / 2)
    pos = {"pre": cent[PRE], "mid": mid, "post": cent[POST]}
    labels = {"pre": f"$t={PRE}$ (before)", "mid": f"$t={MID}$ (occluded)",
              "post": f"$t={POST}$ (after)"}

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.85))
    for j, tag in enumerate(["pre", "mid", "post"]):
        axes[0, j].imshow(sq(f"{A}/rgb_{VID}_{tag}.jpg"))
        axes[1, j].imshow(sq(f"{A}/id_{VID}_{tag}.png"))
        axes[0, j].set_title(labels[tag], fontsize=9, color=INK, pad=3)
        for i in (0, 1):
            c = Circle(pos[tag], 62, fill=False, lw=1.4, ls="--",
                       ec="white" if i else ACC)
            axes[i, j].add_patch(c)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_visible(False)
    axes[0, 0].set_ylabel("input RGB", fontsize=9.5, color=INK)
    axes[1, 0].set_ylabel("generated ID-map", fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.945, bottom=0.01,
                        wspace=0.03, hspace=0.05)
    fig.savefig(f"{F}/fig_reacq.pdf", dpi=300)
    plt.close(fig)
    print("fig_reacq.pdf")


# ---------------------------------------------------------------- fig 4: AssA vs DetA
def fig_scatter():
    # DanceTrack test (DetA, AssA). 2020-22 + 2023-24 VERIFIED against Sun et al. 2022
    # (Table 3) + MOTRv2 (CVPR23); OC-SORT counted by its 2022 arXiv year.
    # 2025-26 group verified from the respective papers' DanceTrack test tables.
    groups = [
        ("trackers 2020–22", "#D1D5DB",
         [("CenterTrack", 78.1, 22.6), ("FairMOT", 66.7, 23.8),
          ("TransTrack", 75.9, 27.5), ("ByteTrack", 71.0, 32.1),
          ("QDTrack", 80.1, 36.8), ("OC-SORT", 80.3, 38.3), ("MOTR", 73.5, 40.2)]),
        ("trackers 2023–24", "#6B7280",
         [("Deep OC-SORT", 82.2, 45.8), ("DiffMOT", 82.5, 47.2),
          ("MeMOTR", 80.5, 58.4), ("MOTRv2", 83.0, 59.0)]),
        ("trackers 2025–26", "white",
         [("SAM2MOT", 79.7, 72.2), ("FDTA", 81.0, 63.5),
          ("MOTIP", 80.4, 60.4), ("SAMURAI++", 82.2, 64.7)]),
    ]
    ours = ("Ours (no detector)", 37.6, 44.1)
    # label anchor (x, y, ha); a thin leader joins the text to its marker
    lab = {"CenterTrack": (78.1, 19.6, "center"), "FairMOT": (64.9, 23.6, "right"),
           "TransTrack": (74.1, 27.3, "right"), "ByteTrack": (69.2, 31.9, "right"),
           "QDTrack": (78.3, 35.6, "right"), "OC-SORT": (82.5, 41.0, "left"),
           "MOTR": (71.7, 40.5, "right"), "Deep OC-SORT": (80.4, 45.3, "right"),
           "DiffMOT": (84.3, 47.0, "left"), "MeMOTR": (78.7, 57.2, "right"),
           "MOTRv2": (84.8, 58.0, "left"), "SAM2MOT": (77.9, 72.4, "right"),
           "FDTA": (78.6, 65.6, "right"), "MOTIP": (78.6, 61.6, "right"),
           "SAMURAI++": (84.8, 66.6, "left")}
    XL, YL = (12, 100), (5, 80)

    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    xs = np.linspace(XL[0], XL[1], 400)
    for H in (30, 40, 50, 60, 70):
        ax.plot(xs, H ** 2 / xs, color=GRID, lw=0.8, ls=(0, (4, 3)), zorder=1)
        yl = H ** 2 / 99
        if YL[0] <= yl <= YL[1] - 4:
            ax.text(99, yl, f"HOTA {H}", fontsize=7, color=MUT,
                    ha="right", va="bottom")
    for gname, fc, pts in groups:
        for name, d, a in pts:
            ax.scatter(d, a, s=38, facecolor=fc, edgecolor=INK, linewidth=0.6,
                       zorder=3, label=gname if name == pts[0][0] else None)
            tx, ty, ha = lab[name]
            ax.annotate(name, (d, a), (tx, ty), ha=ha, va="center", fontsize=7,
                        color=INK, zorder=4,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUT,
                                        shrinkA=1, shrinkB=3.5))
    d, a = ours[1], ours[2]
    ax.scatter(d, a, s=140, marker="*", facecolor=ACC, edgecolor=INK,
               linewidth=0.6, zorder=5, label="ours (no detector)")
    ax.text(d, a + 2.4, ours[0], fontsize=8, color=ACC, weight="bold",
            ha="center", zorder=5)
    ax.plot([22, 68], [22, 68], color=GRID, lw=0.9, zorder=1)
    ax.text(53, 54.3, "AssA = DetA", fontsize=6.5, color=MUT, rotation=45,
            transform_rotates_text=True, ha="center", va="bottom")
    ax.set_xlabel("DetA (detection)")
    ax.set_ylabel("AssA (association)")
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.grid(color="#F0F1F3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=1,
              edgecolor="none", handletextpad=0.3, borderpad=0.4,
              labelspacing=0.25, borderaxespad=0.3)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_assa_deta")


# ---------------------------------------------------------------- fig 5: dense 512 vs 768
def fig_dense():
    panels = [(f"{A}/rgb_0094_f1.jpg", "input RGB (19 people visible)"),
              (f"{A}/gen_0094_f1.png", "generated at $512^2$"),
              (f"{A}/gen768_0094_f1.png", "generated at $768^2$")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    for ax, (p, ttl) in zip(axes, panels):
        ax.imshow(sq(p))
        ax.set_title(ttl, fontsize=9, color=INK, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.02, wspace=0.03)
    fig.savefig(f"{F}/fig_dense.pdf", dpi=300)
    plt.close(fig)
    print("fig_dense.pdf")


# ---------------------------------------------------------------- fig 1: overview diagram
def fig_overview():
    # 10 data units per inch on both axes (thumbnails stay square)
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_xlim(0, 96); ax.set_ylim(-2, 34); ax.axis("off")
    BB = dict(fc="white", ec="none", pad=1.5)  # label background: text never sits on a line

    def box(x, y, w, h, ec=MUT, lw=1.1):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=0.9",
                                    fc="white", ec=ec, lw=lw, zorder=2))

    def txt(x, y, s, fs=9, color=INK, weight="bold", bbox=None):
        ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color,
                weight=weight, zorder=6, bbox=bbox)

    def arrow(p0, p1, label=None, color=MUT, lpos=None):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                     lw=1.3, color=color, zorder=1))
        if label:
            x, y = lpos if lpos else ((p0[0] + p1[0]) / 2, max(p0[1], p1[1]) + 1.2)
            ax.text(x, y, label, ha="center", fontsize=7.5, color=color, zorder=6,
                    bbox=BB)

    def thumb(path, cx, y, w):
        img = sq(path, 256)
        x = cx - w / 2
        ax.imshow(img, extent=(x, x + w, y, y + w), zorder=4, aspect="auto")
        ax.add_patch(mpatches.Rectangle((x, y), w, w, fill=False, ec=MUT, lw=0.7,
                                        zorder=5))

    # input: RGB window (title zone on top, thumbnail below -- no overlap)
    box(1.5, 19, 18.5, 14)
    txt(10.75, 31.2, "RGB window $k$")
    txt(10.75, 29.4, "49 frames, $512^2$", fs=7.5, color=MUT, weight="normal")
    thumb(f"{A}/rgb_0065_f0350.jpg", 10.75, 20.2, 8)

    # input: prefix (cleaned tail) with the explicit 9-frame overlap
    box(1.5, -0.5, 18.5, 16, ec=ACC)
    txt(10.75, 14.0, "cleaned tail of win. $k{-}1$", fs=8.0)
    txt(10.75, 12.4, "9-frame prefix", fs=7.5, color=MUT, weight="normal")
    thumb(f"{A}/id_0065_f0350.png", 10.75, 4.2, 7.5)
    txt(10.75, 2.6, "frames 41–49 of window $k{-}1$", fs=6.5, color=MUT, weight="normal")
    txt(10.75, 1.1, "= frames 1–9 of window $k$", fs=6.5, color=MUT, weight="normal")

    # generator
    box(27, 9.5, 21, 16, ec=ACC, lw=1.7)
    txt(37.5, 22.3, "video diffusion", fs=10)
    txt(37.5, 20.2, "transformer (22B)", fs=10)
    txt(37.5, 17.3, "in-context LoRA ($r{=}32$)", fs=7.5, color=MUT, weight="normal")
    txt(37.5, 15.5, "reference: aligned tokens", fs=7.5, color=MUT, weight="normal")
    txt(37.5, 13.7, "prefix: first 9 frames clean", fs=7.5, color=MUT, weight="normal")

    arrow((20.4, 26), (26.6, 21.5), "reference", lpos=(24.2, 27.3))
    arrow((20.4, 8.5), (26.6, 13.5), "prefix", color=ACC, lpos=(25.3, 7.3))

    # output ID-map window
    box(54, 9.5, 16, 16)
    txt(62, 22.9, "ID-map window $k$", fs=8.8)
    txt(62, 21.1, "colour = identity", fs=7.5, color=MUT, weight="normal")
    thumb(f"{A}/id_0065_f0525.png", 62, 10.6, 8.4)
    arrow((48.4, 17.5), (53.6, 17.5))

    # codec + tracks (right column, pulled in)
    box(76, 20.5, 18, 8)
    txt(85, 26.4, "codec decode")
    txt(85, 24.3, "nearest palette colour", fs=7.5, color=MUT, weight="normal")
    txt(85, 22.6, "+ speckle removal", fs=7.5, color=MUT, weight="normal")
    box(76, 5.5, 18, 8)
    txt(85, 11.4, "tracks")
    txt(85, 9.3, "palette index = track id", fs=7.5, color=MUT, weight="normal")
    txt(85, 7.6, "(no association step)", fs=7.5, color=MUT, weight="normal")
    arrow((70.4, 21.5), (75.6, 25.5))
    arrow((85, 20.1), (85, 13.9))

    # feedback loop routed UNDER the diagram: codec output -> prefix of window k+1.
    # Orthogonal route with rounded corners: down the gap right of the ID-map box,
    # then left below everything, into the prefix box's right edge.
    pts = [(75.6, 21.4), (73.4, 21.4), (73.4, 1.6), (20.4, 1.6)]
    verts, codes, r = [pts[0]], [Path.MOVETO], 1.5
    for a, p, b in zip(pts, pts[1:], pts[2:]):
        ua = np.subtract(a, p) / np.hypot(*np.subtract(a, p))
        ub = np.subtract(b, p) / np.hypot(*np.subtract(b, p))
        verts += [tuple(p + r * ua), p, tuple(p + r * ub)]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.3, color=ACC, zorder=1))
    txt(47, 3.7, "re-encoded cleaned map $\\rightarrow$ prefix for window $k{+}1$",
        fs=8, color=ACC, weight="normal", bbox=BB)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    save(fig, "fig_overview", dpi=300)


# ---------------------------------------------------------------- fig 6: association vs density
def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except ImportError:  # rank correlation by hand, no p-value
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def fig_assa_density():
    """Per-video AssA / DetA of the chained system (Arm A) vs number of GT people."""
    lines = open("paper/data/ltx_arm_a_chain_eval_trackeval_stdout.txt").read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("HOTA:"))
    hdr = lines[i].split()[1:]  # TrackEval fuses the run name onto the first column name
    hdr[0] = "HOTA"
    col = {c: j + 1 for j, c in enumerate(hdr)}  # +1: row token 0 is the video name
    per = {}
    for ln in lines[i + 1:]:
        t = ln.split()
        if not t or t[0] == "COMBINED":
            break
        per[t[0]] = {c: float(t[j]) for c, j in col.items()}
    gt = json.load(open("paper/data/ltx_arm_a_chain_eval_metrics_raw_color.json"))["per_video"]
    vids = sorted(per)
    assert len(vids) == 25 and set(vids) == set(gt), (len(vids), set(vids) ^ set(gt))
    n = np.array([gt[v]["gt_tracks"] for v in vids], float)

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, m in zip(axes, ("AssA", "DetA")):
        y = np.array([per[v][m] for v in vids])
        rho, p = _spearman(n, y)
        k, b = np.polyfit(n, y, 1)
        xx = np.array([n.min() - 1.5, n.max() + 1.5])
        ax.plot(xx, k * xx + b, color=ACC, lw=1.0, alpha=0.35, zorder=1)
        ax.scatter(n, y, s=26, facecolor=ACC, edgecolor="white", linewidth=0.5, zorder=3)
        ptxt = "" if np.isnan(p) else (", $p<0.001$" if p < 1e-3 else f", $p={p:.3f}$")
        ax.text(0.97, 0.95, f"Spearman $\\rho={rho:.2f}$" + ptxt, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK)
        ax.set_xlabel("people in video (GT tracks)")
        ax.set_ylabel(m)
        ax.set_xlim(2, 27)
        ax.grid(color="#F0F1F3", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(f"  {m} vs gt_tracks: Spearman rho={rho:.3f} p={p:.4f} (n={len(vids)})")
    fig.tight_layout(pad=0.4, w_pad=1.5)
    save(fig, "fig_assa_density")


if __name__ == "__main__":
    fig_chain()
    fig_reacq()
    fig_scatter()
    fig_dense()
    fig_overview()
    fig_assa_density()
    print("ALL FIGURES DONE")
