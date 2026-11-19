"""Stage-1 (P0) training data: (frame -> ID-map) pairs from the train split.

Framework-agnostic (numpy in/out) so it's testable without torch; the torch
Dataset wrapper lives in the training script. Frames and ID-maps are produced at
a single square `size` (consistent distortion for both — GT and prediction live in
the same space, so P0 segmentation quality is unaffected; un-distort only for final
pixel-coordinate metrics). ID-maps are rendered directly at the target size by
scaling the GT boxes — no lossy ID-map resize.
"""
from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
from PIL import Image

from gentrack.data.build_idmaps import render_idmap
from gentrack.data.mot_format import Box
from gentrack.data.pseudo_masks import (
    idmap_from_instance_map,
    load_instance_map,
    resize_instance_map,
)
from gentrack.palette import DEFAULT_K, build


def _resize_frame(path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _scale_boxes(boxes, sx: float, sy: float) -> list[Box]:
    return [
        Box(b.frame, b.track_id, b.x * sx, b.y * sy, b.w * sx, b.h * sy, b.conf, b.cls, b.vis)
        for b in boxes
    ]


def stage1_pairs_for_seq(
    seq, palette, size: int, K: int = DEFAULT_K, mask_source: str = "filled_box",
    pseudo_root: str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (frame, idmap) uint8 (size,size,3) pairs for one sequence.

    Stage-1 is single-frame perception → colours are assigned PER FRAME (distinct
    colour per person present), not sequence-stable. This avoids the >K-1-ids
    overflow on long videos (DanceTrack train videos exceed 31 cumulative ids) and
    is sufficient for perception; sequence-stable colours + recycling are a Stage-2
    concern. Frames have <= K-1 simultaneous people on DanceTrack.

    `mask_source="sam_pseudo"` loads the precomputed native-res instance map from
    `{pseudo_root}/{seq.name}/{f:08d}.png`, resizes it NEAREST to the square training
    size, and encodes it (silhouette labels); "filled_box" renders boxes as before.
    """
    w, h = seq.image_size
    sx, sy = size / w, size / h
    for f in range(1, seq.num_frames + 1):
        frame = _resize_frame(seq.frame_path(f), size)
        if mask_source == "sam_pseudo":
            if pseudo_root is None:
                raise ValueError("mask_source='sam_pseudo' requires pseudo_root")
            arr = load_instance_map(os.path.join(pseudo_root, seq.name, f"{f:08d}.png"))
            idmap = idmap_from_instance_map(resize_instance_map(arr, size), palette, K=K)
        else:
            boxes = _scale_boxes(seq.boxes_at(f), sx, sy)
            idmap = render_idmap(boxes, (size, size), palette, assignment=None, K=K,
                                 mask_source=mask_source)
        yield frame, idmap


def iter_stage1(
    sequences, size: int, K: int = DEFAULT_K, palette=None, mask_source: str = "filled_box",
    pseudo_root: str | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (frame, idmap) pairs across all sequences (Stage-1 training stream)."""
    if palette is None:
        palette = build(K)
    for seq in sequences:
        yield from stage1_pairs_for_seq(seq, palette, size, K, mask_source, pseudo_root)
