"""Identity-persistence metrics for P-A (decode-based; §13 GATE A).

The quantitative loop ALWAYS runs through the codec (rule 1 of the working agreement, README.md): generated
ID-maps are decoded to instances, then compared to GT by IoU per identity.

- survival_curve / half_life : how long identities keep the correct colour as the
  chain grows (chain-length test, frame-gap sweep).
- reacquisition_rate         : after an occlusion gap, is the identity re-bound to
  its colour when it reappears (occlusion re-test).

GT here is per-frame {id: bool mask}; `assignment` maps id -> palette index so
decode keys are ids. iou_threshold is an eval parameter pinned in configs/eval.yaml.
"""
from __future__ import annotations

import numpy as np

from gentrack.color_codec import decode


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)


def decode_maps(gen_maps, palette=None, assignment=None, min_area: int = 0):
    return [
        decode(m, palette=palette, assignment=assignment, min_area=min_area)
        for m in gen_maps
    ]


def survival_curve(
    gen_maps,
    gt_per_frame,
    palette=None,
    assignment=None,
    iou_threshold: float = 0.5,
    min_area: int = 0,
) -> list[float]:
    """Fraction of starting identities still *continuously* correct at each frame.

    An identity (present in the first frame) stays "alive" until the first frame
    where GT says it's present but the decoded map fails to recover it at
    IoU >= threshold. Frames where GT marks the id absent (occlusion/exit) do not
    kill it — that is the occlusion test's job (`reacquisition_rate`).
    """
    decoded = decode_maps(gen_maps, palette, assignment, min_area)
    tracked = sorted(gt_per_frame[0].keys()) if gt_per_frame else []
    if not tracked:
        return [0.0 for _ in gen_maps]

    alive = dict.fromkeys(tracked, True)
    curve = []
    for gt_t, dec_t in zip(gt_per_frame, decoded):
        for tid in tracked:
            if not alive[tid] or tid not in gt_t:
                continue
            ok = tid in dec_t and iou(dec_t[tid], gt_t[tid]) >= iou_threshold
            if not ok:
                alive[tid] = False
        curve.append(sum(alive.values()) / len(tracked))
    return curve


def half_life(curve) -> int:
    """First frame index where survival reaches 0.5 or below (half the identities
    lost). Returns len(curve) if it never does (full persistence over the chain).
    """
    for t, v in enumerate(curve):
        if v <= 0.5:
            return t
    return len(curve)


def reacquisition_rate(
    gen_maps,
    gt_per_frame,
    events,
    palette=None,
    assignment=None,
    iou_threshold: float = 0.5,
    min_area: int = 0,
) -> dict:
    """Of mined occlusion events, the fraction whose identity is correctly
    recovered at the reappearance frame (overall + bucketed by gap length).

    `events` are OcclusionEvent objects (data/build_idmaps.py); their
    `after_frame` is 1-indexed (gen_maps[after_frame-1]).
    """
    decoded = decode_maps(gen_maps, palette, assignment, min_area)
    hits, total = 0, 0
    by_gap: dict[int, list[int]] = {}
    for ev in events:
        idx = ev.after_frame - 1
        if idx < 0 or idx >= len(decoded):
            continue
        gt_t, dec_t = gt_per_frame[idx], decoded[idx]
        if ev.track_id not in gt_t:
            continue
        ok = ev.track_id in dec_t and iou(dec_t[ev.track_id], gt_t[ev.track_id]) >= iou_threshold
        hits += int(ok)
        total += 1
        by_gap.setdefault(ev.gap, []).append(int(ok))
    return {
        "rate": hits / total if total else 0.0,
        "hits": hits,
        "total": total,
        "by_gap": {g: sum(v) / len(v) for g, v in sorted(by_gap.items())},
    }
