"""Color codec: lossless mapping between (instance masks + ids) and RGB ID-maps.

The whole approach rests on this round-trip (tests/test_color_codec.py, which
gates the entire project):

    IoU(decode(encode(GT)), GT) >= 0.99  and  100% identity match.

If a downstream metric looks wrong, suspect this path before the model.

Rule 2 of the working agreement (README.md): ID-maps are LOSSLESS PNG only. `save_idmap` /
`load_idmap` refuse any other suffix — JPEG smears flat colours and silently
corrupts decode.

decode strategy: nearest-palette quantisation in CIELAB, then connected
components computed *per colour*. Per-colour CC (rather than CC on the binary
foreground followed by a majority vote) is deliberate: it never merges two
adjacent but differently-coloured instances into one blob, which is the common
case under occlusion/contact. Off-colour speckle inside a region forms its own
tiny components and is removed by `min_area`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from gentrack.palette import DEFAULT_K, _rgb_to_lab, assign, build

_LOSSLESS_SUFFIXES = {".png"}


def _check_lossless(path: Path) -> None:
    if path.suffix.lower() not in _LOSSLESS_SUFFIXES:
        raise ValueError(
            f"ID-maps must be lossless PNG, got {path.suffix!r} ({path})"
        )


def _nearest_palette_index(rgb_uint8: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (H,W) int64 nearest-palette index, distance in CIELAB."""
    lab_img = _rgb_to_lab(rgb_uint8.astype(np.float64) / 255.0)   # (H, W, 3)
    lab_pal = _rgb_to_lab(palette.astype(np.float64) / 255.0)     # (K, 3)
    h, w, _ = lab_img.shape
    flat = lab_img.reshape(-1, 3)                                 # (N, 3)

    best_idx = np.zeros(flat.shape[0], dtype=np.int64)
    best_d2 = np.full(flat.shape[0], np.inf)
    # loop over K colours (cheap) to avoid an (N, K, 3) allocation
    for k in range(lab_pal.shape[0]):
        d2 = np.sum((flat - lab_pal[k]) ** 2, axis=1)
        upd = d2 < best_d2
        best_idx[upd] = k
        best_d2[upd] = d2[upd]
    return best_idx.reshape(h, w)


def encode(masks, ids, K: int = DEFAULT_K, palette=None, assignment=None) -> np.ndarray:
    """Paint instances onto a black RGB ID-map.

    masks: sequence of (H,W) boolean arrays (parallel to `ids`).
    ids:   sequence of integer instance ids.
    Overlaps resolve by paint order — later entries paint over earlier ones, so
    callers should pass masks front-to-back (nearest object last).
    Returns (H, W, 3) uint8.
    """
    masks = [np.asarray(m, dtype=bool) for m in masks]
    ids = list(ids)
    if len(masks) != len(ids):
        raise ValueError(f"masks ({len(masks)}) and ids ({len(ids)}) length mismatch")
    if not masks:
        raise ValueError("encode requires at least one mask")

    if palette is None:
        palette = build(K)
    if assignment is None:
        assignment = assign(ids, K)

    h, w = masks[0].shape
    out = np.zeros((h, w, 3), dtype=np.uint8)  # background = black (index 0)
    for mask, i in zip(masks, ids):
        if mask.shape != (h, w):
            raise ValueError(f"mask shape {mask.shape} != {(h, w)}")
        out[mask] = palette[assignment[i]]
    return out


def decode(
    rgb: np.ndarray,
    K: int = DEFAULT_K,
    palette=None,
    assignment=None,
    min_area: int = 0,
) -> dict[int, np.ndarray]:
    """Recover instance masks from an RGB ID-map.

    Returns {key: (H,W) bool mask}, background excluded. If `assignment`
    (id->index) is given, keys are ids (and unassigned stray colours are
    dropped); otherwise keys are raw palette indices. `min_area` drops connected
    components smaller than that many pixels (speckle removal).
    """
    if palette is None:
        palette = build(K)
    idx_map = _nearest_palette_index(rgb, palette)

    inv = {v: k for k, v in assignment.items()} if assignment is not None else None

    out: dict[int, np.ndarray] = {}
    for c in range(1, len(palette)):
        binary = idx_map == c
        if not binary.any():
            continue
        if min_area > 0:
            labeled, n = ndimage.label(binary)
            if n:
                counts = np.bincount(labeled.ravel())
                small = np.nonzero(counts[1:] < min_area)[0] + 1
                if small.size:
                    binary = binary & ~np.isin(labeled, small)
            if not binary.any():
                continue
        if inv is not None:
            if c not in inv:
                continue  # stray colour with no assigned id
            out[inv[c]] = binary
        else:
            out[c] = binary
    return out


def save_idmap(rgb: np.ndarray, path) -> None:
    """Write an ID-map as lossless PNG. Refuses any lossy suffix."""
    path = Path(path)
    _check_lossless(path)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path, format="PNG")


def load_idmap(path) -> np.ndarray:
    """Read an ID-map PNG -> (H,W,3) uint8. Refuses any lossy suffix."""
    path = Path(path)
    _check_lossless(path)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
