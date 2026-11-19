"""ID-map rendering + stage-1/temporal tuple construction."""
from pathlib import Path

import numpy as np
import pytest

from gentrack.color_codec import decode
from gentrack.data.build_idmaps import (
    filled_box_masks,
    render_idmap,
    sequence_assignment,
    stage1_pairs,
    temporal_triples,
)
from gentrack.data.datasets import Sequence
from gentrack.data.mot_format import Box
from gentrack.palette import assign, build

W, H = 200, 120


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def _seq(n):
    gt = []
    for f in range(1, n + 1):
        gt.append(Box(f, 10, 20, 30, 40, 50))
        gt.append(Box(f, 20, 120, 40, 30, 40))
    frames = [Path(f"{i:06d}.png") for i in range(1, n + 1)]
    return Sequence(
        name="x", root=Path("."), frames=frames, gt=gt,
        info={"name": "x", "imWidth": W, "imHeight": H},
    )


def test_render_idmap_roundtrips():
    boxes = [Box(1, 10, 20, 30, 40, 50), Box(1, 20, 120, 40, 30, 40)]  # disjoint
    pal, a = build(32), assign([10, 20], 32)
    rgb = render_idmap(boxes, (W, H), pal, a)
    assert rgb.shape == (H, W, 3)

    rec = decode(rgb, palette=pal, assignment=a)
    assert set(rec) == {10, 20}
    masks, ids = filled_box_masks(boxes, (W, H))
    gt = {i: m for m, i in zip(masks, ids)}
    for i in (10, 20):
        assert _iou(rec[i], gt[i]) >= 0.99


def test_paint_order_front_on_top():
    front = Box(1, 2, 50, 40, 60, 60, vis=0.9)  # x[50,110] y[40,100]
    back = Box(1, 1, 60, 50, 60, 60, vis=0.2)   # x[60,120] y[50,110]
    pal, a = build(32), assign([1, 2], 32)
    rgb = render_idmap([back, front], (W, H), pal, a)
    rec = decode(rgb, palette=pal, assignment=a)
    # overlap pixel (x=70, y=80) must belong to the higher-visibility front box
    assert rec[2][80, 70]
    assert not rec[1][80, 70]


def test_render_empty_is_black():
    rgb = render_idmap([], (W, H))
    assert rgb.shape == (H, W, 3) and np.all(rgb == 0)


def test_stage1_pairs_count():
    pairs = list(stage1_pairs(_seq(4)))
    assert len(pairs) == 4
    f, p, idmap = pairs[0]
    assert f == 1 and idmap.shape == (H, W, 3)


def test_temporal_triples_stride():
    tr = list(temporal_triples(_seq(5), stride=2))
    assert len(tr) == 3  # t = 1,2,3
    assert tr[0].frame_t == 1 and tr[0].frame_tk == 3 and tr[0].stride == 2
    assert tr[0].idmap_t.shape == (H, W, 3)


def test_sequence_assignment_overflow():
    with pytest.raises(ValueError):
        sequence_assignment(list(range(40)), 32)
