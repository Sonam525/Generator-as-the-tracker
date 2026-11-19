"""CanonicalTracker: decode-free + IoU/colour matching to stable canonical ids."""
import numpy as np

from gentrack.palette import build
from gentrack.pipeline.canonicalize import CanonicalTracker

H, W = 60, 100


def _box(x0, y0, x1, y1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _raw(pairs, palette):
    """Paint (palette_idx, mask) pairs onto black — simulates NB2 free-colour output."""
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, m in pairs:
        out[m] = palette[idx]
    return out


def test_first_frame_creates_distinct_tracks():
    pal = build(32)
    t = CanonicalTracker(pal, min_area=10)
    vis = t.update(_raw([(7, _box(5, 5, 25, 40)), (11, _box(60, 5, 80, 40))], pal))
    assert len(vis) == 2
    assert len(t.tracks) == 2
    assert len(set(t.assignment().values())) == 2  # distinct canonical colours


def test_consecutive_frames_keep_ids():
    pal = build(32)
    t = CanonicalTracker(pal, min_area=10)
    ids1 = set(t.update(_raw([(7, _box(5, 5, 25, 40)), (11, _box(60, 5, 80, 40))], pal)))
    # both shift right 2px, same NB2 colours
    ids2 = set(t.update(_raw([(7, _box(7, 5, 27, 40)), (11, _box(62, 5, 82, 40))], pal)))
    assert ids1 == ids2
    assert len(t.tracks) == 2


def test_position_holds_identity_under_colour_swap():
    pal = build(32)
    t = CanonicalTracker(pal, min_area=10)
    left, right = _box(5, 5, 25, 40), _box(60, 5, 80, 40)
    t.update(_raw([(7, left), (11, right)], pal))
    # same positions, NB2 swapped the two colours — IoU should preserve identity
    t.update(_raw([(11, left), (7, right)], pal))
    assert len(t.tracks) == 2  # no spurious new tracks


def test_reacquisition_by_colour_after_gap():
    pal = build(32)
    t = CanonicalTracker(pal, min_area=10, max_missing=5)
    tid = next(iter(t.update(_raw([(7, _box(5, 5, 25, 40))], pal))))
    t.update(_raw([], pal))  # occluded one frame
    assert t.tracks[tid].missing == 1
    # reappears far away (no IoU overlap) but same colour 7 -> matched by colour
    vis = t.update(_raw([(7, _box(45, 5, 65, 40))], pal))
    assert tid in vis
    assert len(t.tracks) == 1


def test_new_blob_spawns_track():
    pal = build(32)
    t = CanonicalTracker(pal, min_area=10)
    t.update(_raw([(7, _box(5, 5, 25, 40))], pal))
    vis = t.update(_raw([(7, _box(6, 5, 26, 40)), (11, _box(60, 5, 80, 40))], pal))
    assert len(t.tracks) == 2 and len(vis) == 2


def test_track_dropped_after_max_missing():
    pal = build(32)
    t = CanonicalTracker(pal, min_area=10, max_missing=2)
    t.update(_raw([(7, _box(5, 5, 25, 40))], pal))
    for _ in range(3):
        t.update(_raw([], pal))
    assert len(t.tracks) == 0  # aged out
