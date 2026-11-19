"""Pipeline loop test: a scripted mock adapter (no model) drives run_chain and we
check that identity-as-colour survives the decode -> correct -> re-encode chain,
including absorbing per-frame colour noise.
"""
import numpy as np

from gentrack.models.base import GeneratorAdapter
from gentrack.palette import assign, build
from gentrack.pipeline.decode_to_tracks import decode_sequence, mask_to_bbox
from gentrack.pipeline.track import make_composite, reencode, run_chain

H, W = 80, 160


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def _moving_scene(n=4, T=6, step=3, seed=0):
    """n disjoint rectangles translating right by `step` px/frame over T frames."""
    rng = np.random.default_rng(seed)
    ids = [10 + i for i in range(n)]
    base = []
    for i in range(n):
        x0 = 5 + i * 35
        y0 = 15 + int(rng.integers(0, 8))
        base.append((x0, y0, x0 + 25, y0 + 45))
    gt = []
    for t in range(T):
        per = {}
        dx = t * step
        for idv, (x0, y0, x1, y1) in zip(ids, base):
            m = np.zeros((H, W), dtype=bool)
            m[y0:y1, x0 + dx : x1 + dx] = True
            per[idv] = m
        gt.append(per)
    return ids, gt


class _ScriptedAdapter(GeneratorAdapter):
    """Returns the scripted GT ID-map per call, perturbed by colour noise, to
    exercise the loop's codec-based cleaning. Ignores frame/cond content."""

    def __init__(self, idmaps, noise=10, seed=0):
        self._maps = list(idmaps)
        self._i = 0
        self._rng = np.random.default_rng(seed)
        self._noise = noise

    def _next(self):
        m = self._maps[self._i].astype(np.int16)
        self._i += 1
        if self._noise:
            m = m + self._rng.integers(-self._noise, self._noise + 1, m.shape)
        return np.clip(m, 0, 255).astype(np.uint8)

    def perceive(self, frame):
        return self._next()

    def edit(self, frame, cond):
        return self._next()


def test_chain_preserves_identity():
    from gentrack.color_codec import encode

    ids, gt = _moving_scene(n=4, T=6)
    palette = build(32)
    assignment = assign(ids, 32)
    gt_maps = [
        encode(list(g.values()), list(g.keys()), palette=palette, assignment=assignment)
        for g in gt
    ]
    frames = [np.full((H, W, 3), 127, dtype=np.uint8) for _ in gt]

    adapter = _ScriptedAdapter(gt_maps, noise=10)
    clean = run_chain(adapter, frames, palette=palette, assignment=assignment, min_area=4)

    assert len(clean) == len(gt)
    tracks = decode_sequence(clean, palette=palette, assignment=assignment, min_area=4)
    assert set(tracks.keys()) == set(ids)  # identity preserved across the chain
    for tid in ids:
        assert len(tracks[tid]) == len(gt)  # present in every frame
        for t in range(len(gt)):
            assert _iou(tracks[tid][t], gt[t][tid]) >= 0.95


def test_run_chain_with_state_hook():
    """State (MemoryCanvas) integrates into the loop: conditions + gets updated."""
    from gentrack.color_codec import encode
    from gentrack.palette import build
    from gentrack.pipeline.memory_canvas import MemoryCanvas

    ids, gt = _moving_scene(n=3, T=4)
    palette = build(32)
    assignment = assign(ids, 32)
    gt_maps = [
        encode(list(g.values()), list(g.keys()), palette=palette, assignment=assignment)
        for g in gt
    ]
    frames = [np.full((H, W, 3), 127, dtype=np.uint8) for _ in gt]
    canvas = MemoryCanvas(palette=palette, assignment=assignment)

    adapter = _ScriptedAdapter(gt_maps, noise=0)
    clean = run_chain(
        adapter, frames, palette=palette, assignment=assignment, min_area=4, state=canvas
    )
    assert len(clean) == len(gt)
    # the canvas accumulated every identity it saw
    assert set(canvas.tracks.keys()) == set(ids)
    assert all(canvas.tracks[i].best_crop is not None for i in ids)


def test_reencode_empty_is_black():
    out = reencode({}, (H, W), build(32), assign([1], 32))
    assert out.shape == (H, W, 3)
    assert np.all(out == 0)


def test_make_composite_shape_preserved():
    frame = np.full((H, W, 3), 100, dtype=np.uint8)
    idmap = np.full((H, W, 3), 200, dtype=np.uint8)
    comp = make_composite(frame, idmap, alpha=0.4)
    assert comp.shape == (H, W, 3)
    assert comp.dtype == np.uint8
    # 0.6*100 + 0.4*200 = 140
    assert np.all(comp == 140)


def test_mask_to_bbox():
    m = np.zeros((H, W), dtype=bool)
    m[10:30, 40:70] = True
    assert mask_to_bbox(m) == (40, 10, 30, 20)
    assert mask_to_bbox(np.zeros((H, W), dtype=bool)) is None
