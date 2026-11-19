"""chained_track with a mock generator (no torch/model)."""
import numpy as np

from gentrack.color_codec import encode
from gentrack.eval.chained_eval import chained_track, perframe_track
from gentrack.palette import assign, build
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes

H, W = 64, 96


def _mask(x0, y0, x1, y1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class _ScriptedGen:
    """Returns pre-set ID-maps in order (ignores the conditioning image)."""

    def __init__(self, maps):
        self._maps = list(maps)
        self._i = 0

    def generate(self, cond):
        m = self._maps[self._i]
        self._i += 1
        return m


def test_chained_track_canonicalizes_and_is_stable():
    pal = build(48)
    a = assign([7, 9], 48)  # two dancers painted in palette colours 7,9
    gt = [
        {7: _mask(5 + t, 5, 22 + t, 40), 9: _mask(55, 5, 80, 40)} for t in range(5)
    ]
    raw_maps = [encode(list(g.values()), list(g.keys()), palette=pal, assignment=a) for g in gt]

    clean = chained_track(_ScriptedGen(raw_maps), [np.zeros((H, W, 3), np.uint8)] * 5,
                          palette=pal, K=48, min_area=20)
    assert len(clean) == 5
    # canonical tracks are stable: exactly 2 tracks across the whole clip
    boxes = idmaps_to_boxes(clean, palette=pal, min_area=20)
    assert len({b.track_id for b in boxes}) == 2
    # both tracks present in every frame
    per_frame = {}
    for b in boxes:
        per_frame.setdefault(b.frame, set()).add(b.track_id)
    assert all(len(v) == 2 for v in per_frame.values())


def test_perframe_track_canonicalizes_no_copy():
    # perframe_track feeds a black prev map every frame (no copy path) but the
    # CanonicalTracker still threads the two moving dancers into 2 stable tracks.
    pal = build(48)
    a = assign([7, 9], 48)
    gt = [
        {7: _mask(5 + t, 5, 22 + t, 40), 9: _mask(55, 5, 80, 40)} for t in range(5)
    ]
    raw_maps = [encode(list(g.values()), list(g.keys()), palette=pal, assignment=a) for g in gt]

    clean = perframe_track(_ScriptedGen(raw_maps), [np.zeros((H, W, 3), np.uint8)] * 5,
                           palette=pal, K=48, min_area=20)
    assert len(clean) == 5
    boxes = idmaps_to_boxes(clean, palette=pal, min_area=20)
    assert len({b.track_id for b in boxes}) == 2
