# Databricks notebook source
# REVISION CONTROLS requested by the internal ICLR review (2026-09-05). Reuses both arms' published
# cleaned windows (no generation). Everything below is evaluated on the SAME detections and the SAME
# 383 ground-truth gap events as the paper's Tables 2-3:
#  (1) Classical association over the system's OWN per-frame boxes (identical detections, ids dropped):
#      kalman_iou = SORT-style constant-velocity Kalman + Hungarian IoU (motion only);
#      bank_dino  = the project's IdentityBank (appearance DINOv3 CLS + motion, momentum-updated when
#                   visible, frozen under occlusion) with the image-arm's pinned parameters.
#      -> TrackEval HOTA/DetA/AssA/... (AssA compared on identical detections) + re-acquisition k/n.
#  (2) Re-acquisition controls: raw palette-colour agreement across INDEPENDENT windows vs chance (arm B)
#      and colour carry-over during the gap (arm A, chained).
#  (3) Per-window IDF1 and #colours vs window index (error accumulation along the chain, arm A).
#  (4) Same-frame multi-component colours (silent union merges) per arm.
# All parameters pinned here a priori; nothing is tuned against the outcome (working-agreement rule 3, README.md).

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool


_LOGF = None


def log(m):
    """Print and append to RES/run.log (progress is otherwise invisible for notebook tasks)."""
    global _LOGF
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        if _LOGF is None and "RES" in globals():
            _LOGF = open(f"{RES}/run.log", "a")
        if _LOGF is not None:
            _LOGF.write(line + "\n")
            _LOGF.flush()
    except Exception:  # noqa: BLE001
        pass


subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "imageio", "imageio-ffmpeg", "scikit-image", "scipy", "numpy", "motmetrics", "pillow"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=True)  # DINOv3 needs a recent transformers
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")
from gentrack.color_codec import decode as codec_decode  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.data.mot_format import Box  # noqa: E402
from gentrack.eval import mot_metrics  # noqa: E402
from gentrack.models.identity_bank import IdentityBank  # noqa: E402
from gentrack.palette import build  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
def _widget(name, default):
    """One widget, one default. A widget absent from base_parameters raises, and a shared
    try/except would silently revert EVERY parameter to its default: on 2026-09-08 a missing
    `dataset` key made the 5-video seed eval re-run the full 25-video revision under the wrong
    out_name for 3.8 h before the timeout. Silent parameter drift is exactly what the working agreement forbids."""
    try:
        v = dbutils.widgets.get(name)  # noqa: F821
    except Exception:  # noqa: BLE001
        return default
    return v if v not in (None, "") else default


ARMS_IN = _widget("arms", "ltx_arm_a_chain,ltx_arm_b_indep")
TRACKERS_IN = _widget("trackers", "native,kalman_iou,bank_dino")
NPROC = int(_widget("nproc", "4"))
VIDS_IN = _widget("vids", "all25")
OUT_NAME = _widget("out_name", "ltx_revision")
DATASET = _widget("dataset", "dancetrack")  # dancetrack | sportsmot
BOTH_LINKS = _widget("both_links", "0") == "1"   # also score every arm under the other identity link
MECHANISM = _widget("mechanism", "0") == "1"     # pinned carry-over statistics (revision, 2026-11-09)
try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

ALL25 = ["dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
         "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
         "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
         "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
         "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097"]
GT_ROOT = f"{VOL}/{DATASET}/val"
if VIDS_IN.strip().lower() == "all25":
    VIDS = ALL25
elif VIDS_IN.strip().lower() == "all_val":
    VIDS = sorted(os.path.basename(p) for p in glob.glob(f"{GT_ROOT}/*") if os.path.isdir(f"{p}/img1"))
else:
    VIDS = [v.strip() for v in VIDS_IN.split(",") if v.strip()]
ARMS = [a.strip() for a in ARMS_IN.split(",") if a.strip()]
TRACKERS = [t.strip() for t in TRACKERS_IN.split(",") if t.strip()]
NATIVE_LINK = {"ltx_arm_a_chain": "raw_color", "ltx_arm_b_indep": "v3many",
               "ltx_arm_c_chain_ckptA": "raw_color", "ltx_arm_d_indep_ckptB": "v3many"}
RES = f"{VOL}/results/{OUT_NAME}"
os.makedirs(RES, exist_ok=True)
log(f"PARAMS dataset={DATASET} out_name={OUT_NAME} arms={ARMS} trackers={TRACKERS} "
    f"vids={len(VIDS)} nproc={NPROC} both_links={BOTH_LINKS} mechanism={MECHANISM}")   # resolved values on the record before any work starts

# ---- pinned parameters (a priori) ----
K, MIN_AREA, OVERLAP_LINK, SIZE = 48, 150, 0.5, 512
GAP_MIN, CTX, MATCH_IOU, REDETECT_MIN, BORDER_FRAC = 9, 15, 0.3, 3, 0.02          # = ltx_reacq_study
GAP_BINS = [(9, 24, "short9-24"), (25, 48, "subwin25-48"), (49, 150, "xwin49-150"), (151, 10 ** 9, "long>150")]
KAL_IOU, KAL_MAX_AGE, KAL_MIN_HITS = 0.3, 30, 1                                   # SORT defaults, max_age 1.5 s @20fps
BANK_MOMENTUM, BANK_MAX_MISSES, W_APP, W_MOTION, MATCH_THR, BANK_K = 0.9, 50, 0.5, 0.5, 0.6, 5000  # = p2_reid_eval; id space uncapped
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
palette = build(K)


# ---------------------------------------------------------------- decoding (Pool workers)
def iou_t(a, b):
    ix, iy = max(a[0], b[0]), max(a[1], b[1])
    iX = min(a[0] + a[2], b[0] + b[2]); iY = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, iX - ix), max(0.0, iY - iy)
    inter = iw * ih
    uni = a[2] * a[3] + b[2] * b[3] - inter
    return inter / uni if uni > 0 else 0.0


def decode_windows(args):
    """One video of one arm -> per-window {frame: {colour: (box, ncomp)}} + window geometry."""
    arm, vid = args
    meta = json.load(open(f"{VOL}/ltx_ds/fullseq/{vid}_ov9/windows.json"))
    wins = sorted(meta["windows"], key=lambda w: w["idx"])
    win_boxes = []
    for wi, w in enumerate(wins):
        perf = {}
        paths = sorted(glob.glob(f"{VOL}/checkpoints/{arm}/{vid}/cleaned/w{wi:02d}/*.png"))
        for t, p in enumerate(paths, start=1):
            rgb = np.asarray(imageio.imread(p))[:, :, :3]
            masks = codec_decode(rgb, palette=palette, assignment=None, min_area=MIN_AREA)
            d = {}
            for c, m in masks.items():
                ys, xs = np.where(m)
                if xs.size == 0:
                    continue
                box = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
                _, ncomp = ndimage.label(m)
                d[int(c)] = (box, int(ncomp))
            perf[w["start"] + (t - 1)] = d
        win_boxes.append(perf)
    return dict(arm=arm, vid=vid, wins=wins, total=meta["total_frames"], win_boxes=win_boxes)


def assemble(dec, link):
    """Exactly ltx_arm_eval's assembly: owned frames, raw_color or v3many ids, same-frame union.
    Returns pred {frame: [(gid, box)]}, plus per-window ownership and the union-merge counts."""
    wins, win_boxes = dec["wins"], dec["win_boxes"]
    win_colours = [sorted({c for d in perf.values() for c in d}) for perf in win_boxes]
    own_start = [wins[0]["start"]]
    for k in range(1, len(wins)):
        own_start.append((wins[k]["start"] + wins[k - 1]["end"]) // 2 + 1)
    own_end = [own_start[k + 1] - 1 for k in range(len(wins) - 1)] + [wins[-1]["end"]]
    if link == "v3many":
        maps = [dict() for _ in wins]
        nid = 0
        for wi in range(len(wins)):
            cols = win_colours[wi]
            if wi > 0:
                prev = wi - 1
                ov = [f for f in range(wins[wi]["start"], wins[prev]["end"] + 1)
                      if f in win_boxes[wi] and f in win_boxes[prev]]
                pcols = win_colours[prev]
                C = np.zeros((len(cols), len(pcols)))
                for f in ov:
                    cb, pb = win_boxes[wi][f], win_boxes[prev][f]
                    for i, cc in enumerate(cols):
                        if cc in cb:
                            for j, pc in enumerate(pcols):
                                if pc in pb:
                                    C[i, j] += iou_t(cb[cc][0], pb[pc][0])
                if len(cols) and len(pcols):
                    for i, c in enumerate(cols):
                        j = int(np.argmax(C[i]))
                        if C[i, j] > OVERLAP_LINK and pcols[j] in maps[prev]:
                            maps[wi][c] = maps[prev][pcols[j]]
            for c in cols:
                if c not in maps[wi]:
                    maps[wi][c] = nid; nid += 1
        gid_of = lambda wi, c: maps[wi][c]  # noqa: E731
    else:
        gid_of = lambda wi, c: c            # noqa: E731
    acc, multi_comp, frames_with_dets = {}, 0, 0
    for wi in range(len(wins)):
        for gf in range(own_start[wi], own_end[wi] + 1):
            d = win_boxes[wi].get(gf, {})
            if d:
                frames_with_dets += 1
            for c, (b, ncomp) in d.items():
                if ncomp >= 2:
                    multi_comp += 1
                key = (gf, gid_of(wi, c))
                if key in acc:
                    x0 = min(acc[key][0], b[0]); y0 = min(acc[key][1], b[1])
                    x1 = max(acc[key][0] + acc[key][2], b[0] + b[2]); y1 = max(acc[key][1] + acc[key][3], b[1] + b[3])
                    acc[key] = (x0, y0, x1 - x0, y1 - y0)
                else:
                    acc[key] = b
    pred = {}
    for (gf, g), b in acc.items():
        pred.setdefault(gf, []).append((g, b))
    return pred, own_start, own_end, dict(multi_component_boxes=multi_comp, boxes=len(acc), frames=frames_with_dets)


# ---------------------------------------------------------------- trackers over the system's own boxes
def run_kalman_iou(dets_by_frame, total):
    """SORT (Bewley et al. 2016): constant-velocity Kalman on (cx,cy,s,r), Hungarian on IoU."""

    def z(b):
        x, y, w, h = b
        return np.array([x + w / 2, y + h / 2, w * h, w / max(h, 1e-6)], dtype=float)

    def unz(s):
        cx, cy, a, r = float(s[0]), float(s[1]), max(float(s[2]), 1e-6), max(float(s[3]), 1e-6)
        w = np.sqrt(a * r); h = a / w
        return (cx - w / 2, cy - h / 2, w, h)

    F = np.eye(7); F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.zeros((4, 7)); H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
    R = np.eye(4); R[2:, 2:] *= 10.0
    Q = np.eye(7); Q[-1, -1] *= 0.01; Q[4:, 4:] *= 0.01
    P0 = np.eye(7); P0[4:, 4:] *= 1000.0; P0 *= 10.0

    class KT:
        count = 0

        def __init__(self, b):
            self.x = np.zeros(7); self.x[:4] = z(b); self.P = P0.copy()
            KT.count += 1
            self.id = KT.count
            self.age_since_update = 0

        def predict(self):
            if self.x[6] + self.x[2] <= 0:
                self.x[6] = 0.0
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q
            self.age_since_update += 1
            return unz(self.x)

        def update(self, b):
            self.age_since_update = 0
            y = z(b) - H @ self.x
            S = H @ self.P @ H.T + R
            Kg = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + Kg @ y
            self.P = (np.eye(7) - Kg @ H) @ self.P

    tracks, out = [], {}
    for f in range(1, total + 1):
        dets = dets_by_frame.get(f, [])
        preds = [t.predict() for t in tracks]
        matched, um_d = [], set(range(len(dets)))
        if dets and preds:
            C = np.zeros((len(dets), len(preds)))
            for i, d in enumerate(dets):
                for j, p in enumerate(preds):
                    C[i, j] = iou_t(d, p)
            ri, ci = linear_sum_assignment(-C)
            for i, j in zip(ri, ci):
                if C[i, j] >= KAL_IOU:
                    matched.append((i, j)); um_d.discard(i)
        for i, j in matched:
            tracks[j].update(dets[i])
            out.setdefault(f, []).append((tracks[j].id, dets[i]))
        for i in um_d:
            t = KT(dets[i]); tracks.append(t)
            out.setdefault(f, []).append((t.id, dets[i]))
        tracks = [t for t in tracks if t.age_since_update <= KAL_MAX_AGE]
    return out


_dino = _dproc = None


def dino_embed(frame_rgb, boxes):
    """DINOv3 CLS embedding per crop (batched). frame_rgb is the 512x512 squeezed RGB frame."""
    import torch
    from PIL import Image
    global _dino, _dproc
    if _dino is None:
        from transformers import AutoImageProcessor, AutoModel
        _dproc = AutoImageProcessor.from_pretrained(DINO_ID, token=HF_TOKEN)
        _dino = AutoModel.from_pretrained(DINO_ID, token=HF_TOKEN).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        log(f"loaded {DINO_ID} on {next(_dino.parameters()).device}")
    dev = next(_dino.parameters()).device
    crops = []
    for (x, y, w, h) in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(SIZE, int(x + w) + 1), min(SIZE, int(y + h) + 1)
        c = frame_rgb[y0:y1, x0:x1]
        crops.append(Image.fromarray(c if c.size else np.zeros((16, 16, 3), np.uint8)))
    inp = _dproc(images=crops, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = _dino(**inp)
    return o.last_hidden_state[:, 0].float().cpu().numpy()


def run_bank_dino(dets_by_frame, total, seq):
    from gentrack.train.stage1_data import _resize_frame
    bank = IdentityBank(momentum=BANK_MOMENTUM, max_misses=BANK_MAX_MISSES)
    out = {}
    for f in range(1, total + 1):
        dets = dets_by_frame.get(f, [])
        if not dets:
            continue
        frame = np.asarray(_resize_frame(seq.frame_path(f), SIZE))
        embs = dino_embed(frame, dets)
        colors = bank.assign([(e, b) for e, b in zip(embs, dets)], f, w_app=W_APP, w_motion=W_MOTION,
                             match_thr=MATCH_THR, K=BANK_K)
        for b, c in zip(dets, colors):
            if c is not None:
                out.setdefault(f, []).append((int(c), b))
    return out


# ---------------------------------------------------------------- evaluation helpers
def gt_512(seq, total):
    w0, h0 = seq.image_size
    sx, sy = SIZE / w0, SIZE / h0
    tracks = {}
    for f in range(1, total + 1):
        for b in seq.boxes_at(f):
            tracks.setdefault(b.track_id, {})[f] = (b.x * sx, b.y * sy, b.w * sx, b.h * sy)
    return tracks


def mine_events(gt_tracks):
    events = []
    for tid, fb in gt_tracks.items():
        fs = sorted(fb)
        for i in range(1, len(fs)):
            gap = fs[i] - fs[i - 1] - 1
            if gap >= GAP_MIN:
                lb = fb[fs[i - 1]]
                edge = (lb[0] <= BORDER_FRAC * SIZE or lb[1] <= BORDER_FRAC * SIZE or
                        lb[0] + lb[2] >= (1 - BORDER_FRAC) * SIZE or lb[1] + lb[3] >= (1 - BORDER_FRAC) * SIZE)
                events.append(dict(tid=tid, last=fs[i - 1], back=fs[i], gap=gap, kind="exit" if edge else "occlusion"))
    return events


def reacq_events(pred, gt_tracks, events, tag):
    res = []
    for ev in events:
        fb = gt_tracks[ev["tid"]]
        pre_fs = [f for f in sorted(fb) if f <= ev["last"]][-CTX:]
        post_fs = [f for f in sorted(fb) if f >= ev["back"]][:CTX]

        def match_ids(fl):
            ids = []
            for f in fl:
                g = fb[f]
                cands = [(gid, iou_t(p, g)) for gid, p in pred.get(f, [])]
                cands = [c for c in cands if c[1] >= MATCH_IOU]
                if cands:
                    ids.append(max(cands, key=lambda c: c[1])[0])
            return ids
        pre_ids, post_ids = match_ids(pre_fs), match_ids(post_fs)
        tb, rd = len(pre_ids) >= REDETECT_MIN, len(post_ids) >= REDETECT_MIN
        reacq, pre_id, post_id = False, None, None
        if tb and rd:
            pre_id = Counter(pre_ids).most_common(1)[0][0]
            post_id = Counter(post_ids).most_common(1)[0][0]
            reacq = pre_id == post_id
        # controls: chance level (1 / #ids present in the post frames) and carry-over during the gap
        post_ids_present = {gid for f in post_fs for gid, _ in pred.get(f, [])}
        chance = 1.0 / len(post_ids_present) if post_ids_present else None
        gap_ids = {gid for f in range(ev["last"] + 1, ev["back"]) for gid, _ in pred.get(f, [])}
        carry = (pre_id in gap_ids) if pre_id is not None else None
        res.append(dict(tag=tag, **{k: ev[k] for k in ("tid", "gap", "kind")}, tracked_before=tb, redetect=rd,
                        reacq=reacq, chance=chance, pre_id_present_during_gap=carry))
    return res


def mechanism_events(pred, dec, gt_tracks, events, tag):
    """Pinned carry-over statistics (revision pre-registration, 2026-11-09), raw-colour link only,
    for every conditional gap event: how much, how long, where, and through which prefixes the
    pre-gap colour survives the gap. Identity = palette index, so window-level lookups are exact."""
    wins, wb = dec["wins"], dec["win_boxes"]
    out = []
    for ev in events:
        fb = gt_tracks[ev["tid"]]
        pre_fs = [f for f in sorted(fb) if f <= ev["last"]][-CTX:]
        post_fs = [f for f in sorted(fb) if f >= ev["back"]][:CTX]

        def ids_on(fl):
            r = []
            for f in fl:
                c = [(g, iou_t(p, fb[f])) for g, p in pred.get(f, [])]
                c = [x for x in c if x[1] >= MATCH_IOU]
                if c:
                    r.append(max(c, key=lambda x: x[1])[0])
            return r
        pre_ids, post_ids = ids_on(pre_fs), ids_on(post_fs)
        if len(pre_ids) < REDETECT_MIN or len(post_ids) < REDETECT_MIN:
            continue
        pre_id = Counter(pre_ids).most_common(1)[0][0]
        post_id = Counter(post_ids).most_common(1)[0][0]
        lb, nb = fb[ev["last"]], fb[ev["back"]]
        la = max(lb[2] * lb[3], 1.0)
        gap = list(range(ev["last"] + 1, ev["back"]))
        present, areas, iou_int, on_other, ncomps, other_keeps = [], [], [], 0, [], 0
        for f in gap:
            bx = [b for g, b in pred.get(f, []) if g == pre_id]
            if not bx:
                continue
            b = bx[0]
            present.append(f)
            areas.append(b[2] * b[3] / la)
            t = (f - ev["last"]) / (ev["back"] - ev["last"])
            interp = tuple(lb[i] + t * (nb[i] - lb[i]) for i in range(4))
            iou_int.append(iou_t(b, interp))
            others = [(iou_t(b, ob[f]), ob[f]) for tid2, ob in gt_tracks.items() if tid2 != ev["tid"] and f in ob]
            if others and max(o[0] for o in others) >= 0.5:
                on_other += 1
                # POST-HOC EXPLORATORY (added 2026-11-09 after the pinned stats): does that other person keep a
                # colour of its own in this frame (co-located layer), or is it painted only in the carried colour (swap)?
                pbox = max(others, key=lambda o: o[0])[1]
                if any(g != pre_id and iou_t(p, pbox) >= MATCH_IOU for g, p in pred.get(f, [])):
                    other_keeps += 1
            nc = [wb[k][f][pre_id][1] for k in range(len(wins)) if f in wb[k] and pre_id in wb[k][f]]
            if nc:
                ncomps.append(max(nc))
        runs, prev = [], None
        for f in present:
            if runs and f == prev + 1:
                runs[-1] += 1
            else:
                runs.append(1)
            prev = f
        prefixes = []
        for k in range(1, len(wins)):
            s = wins[k]["start"]
            if ev["last"] < s < ev["back"]:
                fr = range(s, s + 9)
                prefixes.append(int(sum(1 for f in fr if f in wb[k - 1] and pre_id in wb[k - 1][f])))
        out.append(dict(tag=tag, tid=ev["tid"], gap=ev["gap"], kind=ev["kind"], reacq=pre_id == post_id,
                        n_gap=len(gap), frac_present=round(len(present) / max(len(gap), 1), 4),
                        longest_run=max(runs) if runs else 0, n_runs=len(runs),
                        rel_area_mean=round(float(np.mean(areas)), 4) if areas else None,
                        iou_interp_mean=round(float(np.mean(iou_int)), 4) if iou_int else None,
                        frac_on_other=round(on_other / len(present), 4) if present else None,
                        frac_other_keeps_own=round(other_keeps / on_other, 4) if on_other else None,
                        ncomp_mean=round(float(np.mean(ncomps)), 3) if ncomps else None,
                        prefix_frames_with_colour=prefixes))
    return out


def per_window_curve(pred, gt_tracks, own_start, own_end):
    """IDF1@0.5 within each window's owned frames + #distinct ids, per window index."""
    rows = []
    for wi, (s, e) in enumerate(zip(own_start, own_end)):
        gt = [Box(f, tid, *fb[f], 1.0, -1, 1.0) for tid, fb in gt_tracks.items() for f in fb if s <= f <= e]
        pr = [Box(f, g, *b, 1.0, -1, 1.0) for f in range(s, e + 1) for g, b in pred.get(f, [])]
        if not gt:
            continue
        m = mot_metrics.evaluate(gt, pr, iou_threshold=0.5) if pr else dict(idf1=0.0)
        rows.append(dict(wi=wi, idf1=round(float(m["idf1"]), 4), ids=len({g for f in range(s, e + 1) for g, _ in pred.get(f, [])}),
                         gt_ids=len({b.track_id for b in gt})))
    return rows


# ---------------------------------------------------------------- TrackEval scaffolding (as ltx_arm_eval)
ROOT = "/local_disk0/te"
BENCH, SPLIT = "GENTRACK", "val"
GTD = f"{ROOT}/gt/mot_challenge/{BENCH}-{SPLIT}"
os.makedirs(f"{ROOT}/gt/mot_challenge/seqmaps", exist_ok=True)


def write_gt(vid, seq, total):
    w0, h0 = seq.image_size
    os.makedirs(f"{GTD}/{vid}/gt", exist_ok=True)
    with open(f"{GTD}/{vid}/gt/gt.txt", "w") as f:
        f.write("\n".join(f"{fr},{b.track_id},{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},1,1,1"
                          for fr in range(1, total + 1) for b in seq.boxes_at(fr)) + "\n")
    with open(f"{GTD}/{vid}/seqinfo.ini", "w") as f:
        f.write(f"[Sequence]\nname={vid}\nimDir=img1\nframeRate=20\nseqLength={total}\nimWidth={w0}\nimHeight={h0}\nimExt=.jpg\n")


def write_pred(tracker, vid, pred, seq):
    w0, h0 = seq.image_size
    ux, uy = w0 / SIZE, h0 / SIZE
    d = f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{tracker}/data"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{vid}.txt", "w") as f:
        f.write("\n".join(f"{fr},{g},{b[0]*ux:.2f},{b[1]*uy:.2f},{b[2]*ux:.2f},{b[3]*uy:.2f},1,-1,-1,-1"
                          for fr in sorted(pred) for g, b in pred[fr]) + "\n")


def run_trackeval(trackers):
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)  # noqa: E731
    if not os.path.exists("/local_disk0/TrackEval"):
        sh("git clone --depth 1 https://github.com/JonathonLuiten/TrackEval /local_disk0/TrackEval")
        sh(r"""find /local_disk0/TrackEval/trackeval -name '*.py' -exec sed -i -E 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.bool\b/bool/g' {} +""")
    r = sh(f"python /local_disk0/TrackEval/scripts/run_mot_challenge.py --GT_FOLDER {ROOT}/gt/mot_challenge "
           f"--TRACKERS_FOLDER {ROOT}/trackers/mot_challenge --BENCHMARK {BENCH} --SPLIT_TO_EVAL {SPLIT} "
           f"--TRACKERS_TO_EVAL {' '.join(trackers)} --METRICS HOTA CLEAR Identity --DO_PREPROC False "
           f"--USE_PARALLEL True --NUM_PARALLEL_CORES {NPROC} --PRINT_CONFIG False --OUTPUT_SUMMARY True "
           f"--OUTPUT_DETAILED True --PLOT_CURVES False")
    open(f"{RES}/trackeval_stdout.txt", "w").write(r.stdout + "\n---STDERR---\n" + r.stderr)
    out = {}
    for t in trackers:
        summ = glob.glob(f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{t}/*summary.txt")
        det = glob.glob(f"{ROOT}/trackers/mot_challenge/{BENCH}-{SPLIT}/{t}/*detailed.csv")
        rec = {}
        if summ:
            lines = [ln.split() for ln in open(summ[0]).read().splitlines() if ln.strip()]
            if len(lines) >= 2:
                rec["combined"] = {k: float(v) for k, v in zip(lines[0], lines[-1]) if v.replace('.', '', 1).replace('-', '', 1).isdigit()}
        if det:
            rec["detailed_csv"] = open(det[0]).read()
        out[t] = rec
    return out


# ---------------------------------------------------------------- main
log(f"dataset={DATASET} arms={ARMS} trackers={TRACKERS} videos={len(VIDS)} nproc={NPROC}")
seqmap = ["name"]
seqs, totals, gts, events_by_vid = {}, {}, {}, {}
for vid in VIDS:
    seqs[vid] = load_sequence(f"{GT_ROOT}/{vid}")
    totals[vid] = json.load(open(f"{VOL}/ltx_ds/fullseq/{vid}_ov9/windows.json"))["total_frames"]
    gts[vid] = gt_512(seqs[vid], totals[vid])
    events_by_vid[vid] = mine_events(gts[vid])
    write_gt(vid, seqs[vid], totals[vid])
    seqmap.append(vid)
open(f"{ROOT}/gt/mot_challenge/seqmaps/{BENCH}-{SPLIT}.txt", "w").write("\n".join(seqmap) + "\n")
log(f"GT ready; gap events: {sum(len(e) for e in events_by_vid.values())}")

all_reacq, curves, merges, tracker_names, mech = [], {}, {}, [], []
for arm in ARMS:
    t0 = time.time()
    with Pool(NPROC) as pool:
        decs = pool.map(decode_windows, [(arm, v) for v in VIDS])
    log(f"{arm}: decoded {len(decs)} videos in {time.time()-t0:.0f}s")
    for dec in decs:
        vid = dec["vid"]
        native_link = NATIVE_LINK.get(arm, "raw_color")
        pred_native, own_s, own_e, mstats = assemble(dec, native_link)
        merges[f"{arm}/{vid}"] = mstats
        dets = {f: [b for _, b in lst] for f, lst in pred_native.items()}
        preds = {}
        if "native" in TRACKERS:
            preds[f"{arm}__native_{native_link}"] = pred_native
        other = "raw_color" if native_link == "v3many" else "v3many"
        if "native" in TRACKERS and (BOTH_LINKS or arm == "ltx_arm_b_indep"):   # the same windows under the other link
            preds[f"{arm}__native_{other}"], _, _, _ = assemble(dec, other)
        if MECHANISM and "native" in TRACKERS:
            praw = pred_native if native_link == "raw_color" else (
                preds.get(f"{arm}__native_raw_color") or assemble(dec, "raw_color")[0])
            mech.extend(dict(vid=vid, **m) for m in mechanism_events(praw, dec, gts[vid], events_by_vid[vid], f"{arm}__native_raw_color"))
        if "kalman_iou" in TRACKERS:
            preds[f"{arm}__kalman_iou"] = run_kalman_iou(dets, dec["total"])
        if "bank_dino" in TRACKERS:
            preds[f"{arm}__bank_dino"] = run_bank_dino(dets, dec["total"], seqs[vid])
        for name, pred in preds.items():
            if name not in tracker_names:
                tracker_names.append(name)
            write_pred(name, vid, pred, seqs[vid])
            all_reacq.extend(dict(vid=vid, **e) for e in reacq_events(pred, gts[vid], events_by_vid[vid], name))
            if name.endswith("native_raw_color") and NATIVE_LINK.get(arm, "raw_color") == "raw_color":
                curves.setdefault(arm, {})[vid] = per_window_curve(pred, gts[vid], own_s, own_e)
        log(f"  {vid}: {len(preds)} trackers done ({time.time()-t0:.0f}s elapsed)")

log("TrackEval over " + ", ".join(tracker_names))
te = run_trackeval(tracker_names)
for t in tracker_names:
    c = te.get(t, {}).get("combined", {})
    log(f"  {t:40s} HOTA {c.get('HOTA')} DetA {c.get('DetA')} AssA {c.get('AssA')} DetRe {c.get('DetRe')} "
        f"DetPr {c.get('DetPr')} AssRe {c.get('AssRe')} AssPr {c.get('AssPr')} IDF1 {c.get('IDF1')} MOTA {c.get('MOTA')} IDSW {c.get('IDSW')}")

# re-acquisition summary per tracker
summary = {}
for t in tracker_names:
    evs = [e for e in all_reacq if e["tag"] == t]
    s = {}
    for lo, hi, lbl in GAP_BINS + [(GAP_MIN, 10 ** 9, "ALL")]:
        sel = [e for e in evs if lo <= e["gap"] <= hi]
        cond = [e for e in sel if e["tracked_before"] and e["redetect"]]
        k = sum(e["reacq"] for e in cond)
        ch = [e["chance"] for e in cond if e["chance"]]
        carry = [e["pre_id_present_during_gap"] for e in cond if e["reacq"] and e["pre_id_present_during_gap"] is not None]
        s[lbl] = dict(events=len(sel), cond_n=len(cond), reacq=k,
                      cond_rate=round(k / len(cond), 3) if cond else None,
                      uncond_rate=round(k / len(sel), 3) if sel else None,
                      mean_chance=round(float(np.mean(ch)), 3) if ch else None,
                      carry_over_frac_of_reacq=round(float(np.mean(carry)), 3) if carry else None)
    summary[t] = s
    log(f"  reacq {t}: " + " | ".join(f"{lbl} {d['reacq']}/{d['cond_n']} ({d['cond_rate']}) unc {d['uncond_rate']} chance {d['mean_chance']} carry {d['carry_over_frac_of_reacq']}" for lbl, d in s.items()))

# per-window curve aggregate (arm A native)
curve = {}
for arm, per_vid in curves.items():
    agg = {}
    for vid, rows in per_vid.items():
        for r in rows:
            agg.setdefault(r["wi"], []).append(r)
    curve[arm] = [dict(wi=wi, n=len(rs), idf1_mean=round(float(np.mean([r["idf1"] for r in rs])), 4),
                       ids_mean=round(float(np.mean([r["ids"] for r in rs])), 2), gt_ids_mean=round(float(np.mean([r["gt_ids"] for r in rs])), 2))
                  for wi, rs in sorted(agg.items())]
    log(f"per-window IDF1 ({arm}): " + ", ".join(f"w{c['wi']}:{c['idf1_mean']:.2f}(n={c['n']})" for c in curve[arm][:40]))

with open(f"{RES}/trackeval_summary.json", "w") as f:
    json.dump({t: te[t].get("combined", {}) for t in tracker_names}, f, indent=1)
with open(f"{RES}/trackeval_detailed.json", "w") as f:
    json.dump({t: te[t].get("detailed_csv", "") for t in tracker_names}, f)
with open(f"{RES}/reacq_events.json", "w") as f:
    json.dump(all_reacq, f)
with open(f"{RES}/reacq_summary.json", "w") as f:
    json.dump(dict(params=dict(gap_min=GAP_MIN, ctx=CTX, match_iou=MATCH_IOU, redetect_min=REDETECT_MIN,
                               kalman=dict(iou=KAL_IOU, max_age=KAL_MAX_AGE, min_hits=KAL_MIN_HITS),
                               bank=dict(momentum=BANK_MOMENTUM, max_misses=BANK_MAX_MISSES, w_app=W_APP,
                                         w_motion=W_MOTION, match_thr=MATCH_THR, dino=DINO_ID)),
                   summary=summary), f, indent=1)
with open(f"{RES}/window_curve.json", "w") as f:
    json.dump(dict(per_video=curves, aggregate=curve), f, indent=1)
with open(f"{RES}/union_merges.json", "w") as f:
    json.dump(merges, f, indent=1)
if MECHANISM:
    with open(f"{RES}/mechanism_events.json", "w") as f:
        json.dump(mech, f)
    log(f"mechanism: {len(mech)} conditional events, {sum(m['reacq'] for m in mech)} re-acquired")
log(f"DONE -> {RES}")
