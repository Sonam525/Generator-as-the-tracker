"""Persistence + re-acquisition metrics."""
import numpy as np

from gentrack.color_codec import encode
from gentrack.data.build_idmaps import OcclusionEvent
from gentrack.eval.persistence import half_life, reacquisition_rate, survival_curve
from gentrack.palette import assign, build

H, W = 60, 100


def _mask(x0, y0, x1, y1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _gt_frames(n):
    gt = []
    for t in range(n):
        gt.append({1: _mask(5 + t, 10, 25 + t, 40), 2: _mask(60, 10, 80, 40)})
    return gt


def _encode_gt(gt, pal, a):
    return [encode(list(g.values()), list(g.keys()), palette=pal, assignment=a) for g in gt]


def test_perfect_survival():
    gt = _gt_frames(8)
    pal, a = build(32), assign([1, 2], 32)
    gen = _encode_gt(gt, pal, a)
    curve = survival_curve(gen, gt, pal, a, iou_threshold=0.5)
    assert curve == [1.0] * 8
    assert half_life(curve) == 8


def test_drift_drops_half_life():
    gt = _gt_frames(8)
    pal, a = build(32), assign([1, 2], 32)
    gen = _encode_gt(gt, pal, a)
    # drop id 1 from frame index 4 onward (only id 2 remains painted)
    for t in range(4, 8):
        gen[t] = encode([gt[t][2]], [2], palette=pal, assignment=a)
    curve = survival_curve(gen, gt, pal, a, 0.5)
    assert curve[:4] == [1.0] * 4
    assert curve[4] == 0.5
    assert half_life(curve) == 4  # half the identities lost at frame 4


def test_reacquisition():
    gt = _gt_frames(8)
    pal, a = build(32), assign([1, 2], 32)
    gen = _encode_gt(gt, pal, a)
    ev = OcclusionEvent(track_id=1, before_frame=2, after_frame=6, gap=3, kind="absence")

    r = reacquisition_rate(gen, gt, [ev], pal, a, 0.5)
    assert r["rate"] == 1.0 and r["total"] == 1 and r["by_gap"] == {3: 1.0}

    # corrupt the reappearance frame (index 5 = frame 6): id 1 not recovered
    gen[5] = encode([gt[5][2]], [2], palette=pal, assignment=a)
    r2 = reacquisition_rate(gen, gt, [ev], pal, a, 0.5)
    assert r2["rate"] == 0.0
