"""Palette + id->colour assignment. Index 0 = background (black).

FROZEN ZONE: the palette and assignment logic live ONLY here.
Changing either re-runs the full invariant suite and invalidates cross-run
comparability — say so in the experiment log before touching it.

The palette is K maximally-distinct colours chosen by farthest-point sampling
in CIELAB (perceptually uniform), so that a generator's *approximate* colour
output still quantises back to the right identity in `color_codec.decode`.
"""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

BACKGROUND_INDEX = 0
# K=48 (was 32): DanceTrack group dances reach 40 simultaneous people; 31 colours
# couldn't represent them. Changed with human sign-off 2026-06-12 (frozen-zone change;
# codec invariant re-verified). Metrics are colour-agnostic so prior verdicts stand.
DEFAULT_K = 48

# Candidate cube: _GRID_LEVELS**3 RGB colours to sample the palette from.
# 8 levels -> 512 candidates: dense enough for a well-spread K<=32 palette.
_GRID_LEVELS = 8


def _rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB floats in [0,1], shape (..., 3) -> CIELAB, same leading shape."""
    return rgb2lab(rgb01)


def _candidate_cube() -> np.ndarray:
    """(_GRID_LEVELS**3, 3) uint8 grid of RGB candidate colours."""
    levels = np.linspace(0, 255, _GRID_LEVELS).round().astype(np.uint8)
    r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, 3).astype(np.uint8)


def build(K: int = DEFAULT_K) -> np.ndarray:
    """Return a deterministic (K, 3) uint8 palette.

    Row 0 is black (background). Rows 1..K-1 are farthest-point-sampled in
    CIELAB to maximise the minimum pairwise perceptual distance. Deterministic:
    ties in the greedy pick break to the lowest candidate index.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    cand_rgb = _candidate_cube()
    cand_lab = _rgb_to_lab(cand_rgb.astype(np.float64) / 255.0)  # (M, 3)
    black_lab = _rgb_to_lab(np.zeros((1, 3)))                    # (1, 3)

    # min squared CIELAB distance from each candidate to the selected set,
    # which starts as {black} so picks spread away from the background too.
    min_d2 = np.sum((cand_lab - black_lab) ** 2, axis=1)

    selected = [np.zeros(3, dtype=np.uint8)]  # index 0 = black
    for _ in range(K - 1):
        pick = int(np.argmax(min_d2))
        selected.append(cand_rgb[pick].copy())
        d2 = np.sum((cand_lab - cand_lab[pick]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[pick] = -1.0  # never pick the same candidate twice

    return np.stack(selected, axis=0).astype(np.uint8)


def assign(ids, K: int = DEFAULT_K) -> dict[int, int]:
    """Stable id -> palette-index map (1..K-1).

    Distinct ids are sorted and assigned in order, so the mapping is stable for
    a given set of ids. Raises rather than silently colliding colours when there
    are more distinct ids than available palette slots (K-1) — colour recycling
    across a long sequence is the pipeline's job, not the codec's.
    """
    uniq = sorted({int(i) for i in ids})
    if len(uniq) > K - 1:
        raise ValueError(
            f"{len(uniq)} distinct ids exceed K-1={K - 1} available colours"
        )
    return {i: n + 1 for n, i in enumerate(uniq)}
