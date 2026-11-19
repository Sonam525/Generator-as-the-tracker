"""Palette invariants: distinctness, determinism, safe id assignment."""
import numpy as np
import pytest
from skimage.color import rgb2lab

from gentrack.palette import BACKGROUND_INDEX, assign, build


def _min_pairwise_lab(palette):
    lab = rgb2lab(palette.astype(np.float64) / 255.0)  # (K, 3)
    d2 = np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def test_shape_and_background():
    pal = build(32)
    assert pal.shape == (32, 3)
    assert pal.dtype == np.uint8
    assert np.array_equal(pal[BACKGROUND_INDEX], [0, 0, 0])


@pytest.mark.parametrize("k", [32, 48])
def test_colours_are_distinct(k):
    pal = build(k)
    # all rows unique
    assert len({tuple(c) for c in pal}) == k
    # and perceptually well separated (Delta-E well above quantisation noise)
    assert _min_pairwise_lab(pal) > 15.0


def test_deterministic():
    assert np.array_equal(build(32), build(32))


@pytest.mark.parametrize("k", [2, 8, 16, 32, 48])
def test_various_k(k):
    pal = build(k)
    assert pal.shape == (k, 3)
    assert np.array_equal(pal[0], [0, 0, 0])


def test_assign_stable_and_in_range():
    ids = [7, 3, 100, 3, 7]  # duplicates + non-contiguous
    a = assign(ids, 32)
    assert set(a.keys()) == {3, 7, 100}
    assert sorted(a.values()) == [1, 2, 3]      # sorted ids -> 1..n
    assert a[3] == 1 and a[7] == 2 and a[100] == 3
    assert assign(ids, 32) == a                  # stable


def test_assign_refuses_overflow():
    with pytest.raises(ValueError):
        assign(range(40), 32)  # 40 distinct ids > 31 slots
