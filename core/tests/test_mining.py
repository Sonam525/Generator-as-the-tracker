"""Occlusion-triple mining (§8.3): DanceTrack absence + MOT17 low-visibility."""
import pytest

from gentrack.data.build_idmaps import mine_absence, mine_low_visibility, mine_occlusion
from gentrack.data.mot_format import Box


def _track(tid, frames, vis=None):
    return [
        Box(f, tid, 0, 0, 10, 10, vis=(vis[k] if vis is not None else 1.0))
        for k, f in enumerate(frames)
    ]


def test_mine_absence_gap_in_range():
    frames = list(range(1, 6)) + list(range(16, 21))  # present 1-5, 16-20 -> gap 10
    ev = mine_absence(_track(1, frames), min_gap=5, max_gap=40)
    assert len(ev) == 1
    e = ev[0]
    assert (e.before_frame, e.after_frame, e.gap, e.kind) == (5, 16, 10, "absence")


def test_mine_absence_out_of_range():
    assert mine_absence(_track(1, [1, 2, 3, 6]), min_gap=5, max_gap=40) == []   # gap 2
    assert mine_absence(_track(1, [1, 60]), min_gap=5, max_gap=40) == []        # gap 58


def test_mine_low_visibility_run():
    frames = list(range(1, 10))
    vis = [1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1]  # low run frames 3-7 (len 5)
    ev = mine_low_visibility(_track(1, frames, vis), vis_threshold=0.5, min_gap=5, max_gap=40)
    assert len(ev) == 1
    e = ev[0]
    assert (e.before_frame, e.after_frame, e.gap, e.kind) == (2, 8, 5, "low_visibility")


def test_mine_low_visibility_needs_both_flanks():
    frames = list(range(1, 8))
    vis = [0.1, 0.1, 0.1, 0.1, 0.1, 1, 1]  # low run starts at frame 1: no "before"
    assert mine_low_visibility(_track(1, frames, vis), 0.5, 5, 40) == []


def test_dispatch_and_unknown():
    frames = list(range(1, 6)) + list(range(16, 21))
    assert len(mine_occlusion(_track(1, frames), "absence", min_gap=5, max_gap=40)) == 1
    with pytest.raises(ValueError):
        mine_occlusion([], "bogus")
