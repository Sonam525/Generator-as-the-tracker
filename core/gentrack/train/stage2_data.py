"""Stage-2 (P1) temporal training data: consecutive (frame, ID-map) pairs with
SEQUENCE-STABLE colours — the same dancer keeps one colour across the whole video,
so the model can learn to PROPAGATE identity (the temporal-binding task).

Yields (frame_t, idmap_t, frame_{t+k}, idmap_{t+k}). The trainer conditions on a
composite of (frame_{t+k}, idmap_t) and targets idmap_{t+k} — "given the new frame
and where the colours were, paint the new ID-map keeping each colour on its dancer".

Sequence-stable colours via `sequence_assignment` (fixed per id); valid because max
cumulative ids/train-video = 40 ≤ K-1=47 (no recycling needed). Strided variants
(k>1) teach robustness to frame gaps.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from gentrack.data.build_idmaps import render_idmap, sequence_assignment
from gentrack.palette import DEFAULT_K, build
from gentrack.train.stage1_data import _resize_frame, _scale_boxes


def stage2_pairs_for_seq(
    seq, palette, size: int, stride: int = 1, K: int = DEFAULT_K, mask_source: str = "filled_box"
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (frame_t, idmap_t, frame_t+stride, idmap_t+stride), sequence-stable colours."""
    w, h = seq.image_size
    sx, sy = size / w, size / h
    assignment = sequence_assignment(seq.track_ids, K)  # one fixed colour per dancer

    def idmap_at(f):
        return render_idmap(_scale_boxes(seq.boxes_at(f), sx, sy), (size, size),
                            palette, assignment, mask_source=mask_source)

    for f in range(1, seq.num_frames - stride + 1):
        yield (_resize_frame(seq.frame_path(f), size), idmap_at(f),
               _resize_frame(seq.frame_path(f + stride), size), idmap_at(f + stride))


def iter_stage2(
    sequences, size: int, strides=(1,), K: int = DEFAULT_K, palette=None, mask_source: str = "filled_box"
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield temporal pairs across sequences and strides (Stage-2 stream)."""
    if palette is None:
        palette = build(K)
    for seq in sequences:
        for s in strides:
            yield from stage2_pairs_for_seq(seq, palette, size, s, K, mask_source)
