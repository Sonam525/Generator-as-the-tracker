"""THE round-trip invariant — gates the whole project (rule 1 of the working agreement, README.md):

    IoU(decode(encode(GT)), GT) >= 0.99  and  100% identity match.

Also covers: lossless PNG round-trip, robustness to colour noise (the realistic
case where a generator emits approximate colours), and the lossy-format guard.
"""
import numpy as np
import pytest

from gentrack.color_codec import decode, encode, load_idmap, save_idmap
from gentrack.palette import assign, build

H, W = 128, 160
IOU_GATE = 0.99


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def _make_scene(n, seed=0):
    """n disjoint rectangle instances on a grid, with non-contiguous ids."""
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    ch, cw = H // rows, W // cols
    masks, ids = [], []
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= n:
                break
            pad_y, pad_x = max(2, ch // 5), max(2, cw // 5)
            y0 = r * ch + rng.integers(2, pad_y + 1)
            x0 = c * cw + rng.integers(2, pad_x + 1)
            y1 = r * ch + ch - rng.integers(2, pad_y + 1)
            x1 = c * cw + cw - rng.integers(2, pad_x + 1)
            m = np.zeros((H, W), dtype=bool)
            m[y0:y1, x0:x1] = True
            masks.append(m)
            ids.append(101 + k * 7)  # arbitrary, non-contiguous ids
            k += 1
    return masks, ids


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 5, 12, 31])
def test_roundtrip_invariant(seed, n):
    masks, ids = _make_scene(n, seed=seed)
    palette = build(32)
    assignment = assign(ids, 32)

    rgb = encode(masks, ids, palette=palette, assignment=assignment)
    rec = decode(rgb, palette=palette, assignment=assignment)

    assert set(rec.keys()) == set(ids)  # 100% identity match
    for mask, i in zip(masks, ids):
        assert _iou(rec[i], mask) >= IOU_GATE, f"id {i}: IoU {_iou(rec[i], mask):.4f}"


def test_roundtrip_invariant_k48():
    """K=48 (DanceTrack dense frames up to 40 people) round-trips losslessly."""
    masks, ids = _make_scene(40, seed=0)
    palette = build(48)
    assignment = assign(ids, 48)
    rgb = encode(masks, ids, palette=palette, assignment=assignment)
    rec = decode(rgb, palette=palette, assignment=assignment)
    assert set(rec.keys()) == set(ids)
    for mask, i in zip(masks, ids):
        assert _iou(rec[i], mask) >= IOU_GATE


def test_background_is_black():
    masks, ids = _make_scene(6, seed=0)
    rgb = encode(masks, ids)
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        union |= m
    assert np.all(rgb[~union] == 0)


def test_png_roundtrip_is_lossless(tmp_path):
    masks, ids = _make_scene(8, seed=2)
    palette = build(32)
    assignment = assign(ids, 32)

    rgb = encode(masks, ids, palette=palette, assignment=assignment)
    p = tmp_path / "idmap.png"
    save_idmap(rgb, p)
    loaded = load_idmap(p)

    assert np.array_equal(loaded, rgb)  # PNG is bit-exact
    rec = decode(loaded, palette=palette, assignment=assignment)
    assert set(rec.keys()) == set(ids)
    for mask, i in zip(masks, ids):
        assert _iou(rec[i], mask) >= IOU_GATE


def test_robust_to_colour_noise():
    """Generator output is approximate; CIELAB nearest-palette should absorb it."""
    masks, ids = _make_scene(10, seed=3)
    palette = build(32)
    assignment = assign(ids, 32)

    rgb = encode(masks, ids, palette=palette, assignment=assignment).astype(np.int16)
    rng = np.random.default_rng(0)
    noisy = np.clip(rgb + rng.integers(-12, 13, size=rgb.shape), 0, 255).astype(np.uint8)

    rec = decode(noisy, palette=palette, assignment=assignment, min_area=4)
    assert set(rec.keys()) == set(ids)
    for mask, i in zip(masks, ids):
        assert _iou(rec[i], mask) >= 0.97


def test_save_refuses_lossy(tmp_path):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        save_idmap(rgb, tmp_path / "bad.jpg")
    with pytest.raises(ValueError):
        load_idmap(tmp_path / "bad.jpeg")
