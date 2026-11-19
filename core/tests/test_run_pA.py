"""P-A runner core: spend gate + chained tracking + gap sweep, with an oracle
adapter (no network, no spend). Frames ARE the GT ID-maps, so a perfect adapter
just echoes its input → identity survives fully."""
from pathlib import Path

import numpy as np
import pytest

from gentrack.color_codec import encode
from gentrack.data.datasets import Sequence
from gentrack.data.mot_format import Box
from gentrack.eval.run_pA import (
    SpendNotConfirmed,
    confirm_spend,
    gap_sweep,
    run_pa_clip,
    sequence_gt_masks,
    track_sequence,
)
from gentrack.models.base import GeneratorAdapter
from gentrack.palette import assign, build

H, W = 50, 90


def _mask(x0, y0, x1, y1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _scene(n):
    return [{1: _mask(5 + t, 5, 20 + t, 30), 2: _mask(55, 5, 75, 30)} for t in range(n)]


def _frames(gt, pal, a):
    return [encode(list(g.values()), list(g.keys()), palette=pal, assignment=a) for g in gt]


class _Oracle(GeneratorAdapter):
    """Echoes the input frame (which, in tests, IS the GT ID-map)."""

    def perceive(self, frame):
        return frame.copy()

    def edit(self, frame, cond):
        return frame.copy()


def test_track_sequence_perfect():
    gt = _scene(6)
    pal, a = build(32), assign([1, 2], 32)
    frames = _frames(gt, pal, a)
    r = track_sequence(_Oracle(), frames, gt, palette=pal, assignment=a, min_area=0)
    assert r["survival_curve"] == [1.0] * 6
    assert r["half_life"] == 6


def test_run_pa_clip_stable_tracks():
    """Canonical-tracker clip path: stable input → stable track count, no churn."""
    gt = _scene(6)
    pal, a = build(32), assign([1, 2], 32)
    frames = _frames(gt, pal, a)  # frames ARE the GT ID-maps (oracle echoes them)
    res = run_pa_clip(_Oracle(), frames, gt, palette=pal, K=32, min_area=10, use_canvas=True)
    assert res["n_tracks"] == 2  # 2 dancers, no spurious tracks
    assert len(res["clean_maps"]) == 6


def test_gap_sweep():
    gt = _scene(8)
    pal, a = build(32), assign([1, 2], 32)
    frames = _frames(gt, pal, a)
    sweep = gap_sweep(_Oracle(), frames, gt, strides=[1, 2, 4], palette=pal, assignment=a, min_area=0)
    assert set(sweep) == {1, 2, 4}
    for s in (1, 2, 4):
        assert all(v == 1.0 for v in sweep[s]["survival_curve"])


def test_contact_sheets_written(tmp_path):
    gt = _scene(4)
    pal, a = build(32), assign([1, 2], 32)
    frames = _frames(gt, pal, a)
    r = track_sequence(
        _Oracle(), frames, gt, palette=pal, assignment=a, min_area=0,
        contact_dir=tmp_path, contact_every=2,
    )
    assert len(r["contact_sheets"]) == 2  # 4 frames / 2 per sheet
    assert all(Path(p).exists() for p in r["contact_sheets"])


def test_confirm_spend():
    assert confirm_spend(10, confirm_above=10.0) == pytest.approx(1.5)  # under gate
    with pytest.raises(SpendNotConfirmed):
        confirm_spend(1000, confirm_above=10.0)  # $150 > $10
    assert confirm_spend(1000, confirm_above=10.0, assume_yes=True) == pytest.approx(150.0)


def test_sequence_gt_masks():
    gt = [Box(1, 1, 5, 5, 10, 10), Box(1, 2, 30, 5, 10, 10), Box(2, 1, 6, 5, 10, 10)]
    seq = Sequence(
        "x", Path("."), [Path("1.png"), Path("2.png")], gt,
        {"name": "x", "imWidth": W, "imHeight": H},
    )
    masks = sequence_gt_masks(seq)
    assert len(masks) == 2
    assert set(masks[0]) == {1, 2}
    assert set(masks[1]) == {1}
