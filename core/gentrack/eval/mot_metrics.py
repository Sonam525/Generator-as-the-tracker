"""Standard MOT identity metrics via motmetrics (box-IoU): IDF1/IDP/IDR, MOTA,
MOTP, ID-switches, MT/ML, FP/FN. These are the §9 identity metrics for local val.

HOTA/AssA (the DanceTrack-leaderboard primary) come from TrackEval at P4 for
anchor comparability — heavier, set up separately; not needed for P-A iteration.

GT here is the dataset's actual boxes; predictions are bboxes of decoded silhouettes
(decode_to_tracks.idmaps_to_boxes), so matching is box-IoU — the like-for-like
comparison (mask-IoU vs filled boxes would be unfairly pessimistic).
"""
from __future__ import annotations

import numpy as np

# motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2.0. Shim it before import.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # noqa: NPY201

import motmetrics as mm  # noqa: E402

from gentrack.data.mot_format import group_by_frame  # noqa: E402

_METRICS = [
    "idf1", "idp", "idr", "mota", "motp",
    "num_switches", "mostly_tracked", "mostly_lost",
    "num_false_positives", "num_misses", "num_objects", "num_unique_objects",
]


def _xywh(boxes):
    return np.array([[b.x, b.y, b.w, b.h] for b in boxes], dtype=float) if boxes else np.empty((0, 4))


def evaluate(gt_boxes, pred_boxes, iou_threshold: float = 0.5) -> dict:
    """IDF1/MOTA/IDSW etc. for one sequence. `iou_threshold` is the min box-IoU
    for a match (0.5 = MOTChallenge standard)."""
    acc = mm.MOTAccumulator(auto_id=False)
    gt_bf, pred_bf = group_by_frame(gt_boxes), group_by_frame(pred_boxes)
    max_dist = 1.0 - iou_threshold  # motmetrics distance = 1 - IoU; pairs above are non-matches
    for f in sorted(set(gt_bf) | set(pred_bf)):
        g, p = gt_bf.get(f, []), pred_bf.get(f, [])
        dist = mm.distances.iou_matrix(_xywh(g), _xywh(p), max_iou=max_dist)
        acc.update([b.track_id for b in g], [b.track_id for b in p], dist, frameid=f)

    summary = mm.metrics.create().compute(acc, metrics=_METRICS, name="seq")
    row = summary.iloc[0].to_dict()
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}


def _box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _pred_id_for(gt_box, preds, iou_thr):
    best_id, best = None, iou_thr
    for p in preds:
        i = _box_iou(gt_box.xywh, p.xywh)
        if i >= best:
            best, best_id = i, p.track_id
    return best_id


def detection_recall(gt_boxes, pred_boxes, iou_thrs=(0.5, 0.3, 0.1)) -> dict:
    """Identity-free detection recall: per IoU threshold, the fraction of GT boxes
    that have ANY pred box (any id) overlapping at >= thr in the same frame.

    Decouples localisation from association — measures whether the perceiver puts a
    box on each object at all, which upper-bounds IDF1 (you can't associate what you
    never detect). The probe's go/no-go input for whether a memory mechanism has
    anything to build on.
    """
    gt_bf, pred_bf = group_by_frame(gt_boxes), group_by_frame(pred_boxes)
    total = sum(len(v) for v in gt_bf.values())
    out = {"n_gt": total}
    for thr in iou_thrs:
        hit = 0
        for f, gs in gt_bf.items():
            ps = pred_bf.get(f, [])
            for g in gs:
                if any(_box_iou(g.xywh, p.xywh) >= thr for p in ps):
                    hit += 1
        out[f"recall@{thr}"] = round(hit / total, 4) if total else 0.0
    return out


def reacquisition_check(pred_boxes, gt_boxes, events, iou_thr: float = 0.5) -> dict:
    """Per occlusion event (GT track G, before/after frames): did the prediction
    re-bind G to the SAME track id after the gap?

    - redetected: G is matched by some pred box at the reappearance frame.
    - reacquired: that pred id equals the one matched to G just before the gap
      (i.e. no identity switch across the occlusion — the Option-A win condition).
    """
    gt_bf, pred_bf = group_by_frame(gt_boxes), group_by_frame(pred_boxes)
    results = []
    for e in events:
        gb = [b for b in gt_bf.get(e.before_frame, []) if b.track_id == e.track_id]
        ga = [b for b in gt_bf.get(e.after_frame, []) if b.track_id == e.track_id]
        if not gb or not ga:
            continue
        pb = _pred_id_for(gb[0], pred_bf.get(e.before_frame, []), iou_thr)
        pa = _pred_id_for(ga[0], pred_bf.get(e.after_frame, []), iou_thr)
        results.append({
            "track": e.track_id, "gap": e.gap,
            "before_frame": e.before_frame, "after_frame": e.after_frame,
            "pred_before": pb, "pred_after": pa,
            "redetected": pa is not None,
            "reacquired": pb is not None and pa is not None and pb == pa,
        })
    n = len(results)
    return {
        "events": results,
        "n": n,
        "redetect_rate": sum(r["redetected"] for r in results) / n if n else 0.0,
        "reacq_rate": sum(r["reacquired"] for r in results) / n if n else 0.0,
    }
