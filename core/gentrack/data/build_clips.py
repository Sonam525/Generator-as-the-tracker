"""Clip construction for the video (Option-C / P2-C) arm: contiguous windows of
(frame, ID-map) with SEQUENCE-STABLE colours — the v2v target is an aligned ID-map
clip where each identity keeps one colour across the whole window (and, via overlap
conditioning, across windows).

This is the video analog of train/stage2_data.py: same sequence-stable colours
(`sequence_assignment`), same mask sources (filled_box | sam_pseudo), same square-resize
convention (_resize_frame / _scale_boxes). The unit of supervision is a length-L clip,
not a pair.

OCCLUSION CLIPS are windows positioned to CONTAIN a mined occlusion event
(visible -> occluded -> reappears) so the gap falls INSIDE the window — that is the
P2-C bet: within a window, temporal attention can carry identity across the gap (the
re-acquisition the appearance-only bank could not do). Gaps longer than the window are
a cross-window concern (overlap conditioning / a bank on top), not handled here.

Pure/numpy (no torch) so it is CPU-testable. How a clip is packed into LTX's latent
layout (channel-concat vs temporal-pack) is the adapter/probe's job — this module is
model-agnostic and knows nothing about LTX's frame-count constraint (the config pins a
length valid for the model; see models.ltx.valid_ltx_length).
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Sequence as Seq
from dataclasses import dataclass

import numpy as np

from gentrack.data.build_idmaps import OcclusionEvent, mine_occlusion, render_idmap, sequence_assignment
from gentrack.data.pseudo_masks import (
    idmap_from_instance_map,
    load_instance_map,
    resize_instance_map,
)
from gentrack.palette import DEFAULT_K, build
from gentrack.train.stage1_data import _resize_frame, _scale_boxes


@dataclass
class ClipSpec:
    """A window of frame indices over one sequence (1-indexed, in capture order)."""
    seq_name: str
    frames: list[int]
    kind: str = "window"                 # "window" | "occlusion"
    event: OcclusionEvent | None = None  # the mined event, for occlusion clips


def clip_starts(num_frames: int, length: int, hop: int, frame_stride: int = 1) -> list[int]:
    """Start frames (1-indexed) of sliding windows that fit fully inside the sequence.

    A window of `length` frames at `frame_stride` spans `(length-1)*frame_stride + 1`
    frames; it must end at or before `num_frames`. Windows step by `hop` start-frames
    (hop == length => non-overlapping). Returns [] if even one window does not fit.
    """
    if length < 1 or hop < 1 or frame_stride < 1:
        raise ValueError("length, hop, frame_stride must be >= 1")
    last = num_frames - (length - 1) * frame_stride
    return list(range(1, last + 1, hop)) if last >= 1 else []


def clip_frames(start: int, length: int, frame_stride: int = 1) -> list[int]:
    """The frame indices of one window."""
    return [start + i * frame_stride for i in range(length)]


def window_specs(seq, length: int, hop: int, frame_stride: int = 1) -> Iterator[ClipSpec]:
    """Sliding-window ClipSpecs over a sequence (the bulk training stream)."""
    for s in clip_starts(seq.num_frames, length, hop, frame_stride):
        yield ClipSpec(seq.name, clip_frames(s, length, frame_stride), "window")


def occlusion_specs(
    seq, length: int, *, frame_stride: int = 1,
    strategy: str = "absence", min_gap: int = 5, max_gap: int = 40, vis_threshold: float = 0.5,
) -> Iterator[ClipSpec]:
    """Windows centred on mined occlusion events, so each event's [before, after] span
    lands inside the window. Events whose span exceeds the window cover are skipped
    (the gap cannot fit — a cross-window case)."""
    cover = (length - 1) * frame_stride + 1
    max_start = seq.num_frames - cover + 1
    if max_start < 1:
        return
    for ev in mine_occlusion(seq.gt, strategy, min_gap=min_gap, max_gap=max_gap,
                             vis_threshold=vis_threshold):
        span = ev.after_frame - ev.before_frame + 1
        if span > cover:
            continue  # gap too long to sit inside one window
        mid = (ev.before_frame + ev.after_frame) // 2
        start = mid - cover // 2
        start = max(1, min(start, max_start))
        # frame_stride may not land exactly on before/after; the window still covers them.
        yield ClipSpec(seq.name, clip_frames(start, length, frame_stride), "occlusion", ev)


def _idmap_at(seq, f: int, size: int, palette, assignment, mask_source: str,
              pseudo_root: str | None, sx: float, sy: float, K: int) -> np.ndarray:
    """One frame's ID-map at the square training size, sequence-stable colours."""
    if mask_source == "sam_pseudo":
        if pseudo_root is None:
            raise ValueError("mask_source='sam_pseudo' requires pseudo_root")
        arr = load_instance_map(os.path.join(pseudo_root, seq.name, f"{f:08d}.png"))
        return idmap_from_instance_map(resize_instance_map(arr, size), palette,
                                       assignment=assignment, K=K)
    boxes = _scale_boxes(seq.boxes_at(f), sx, sy)
    return render_idmap(boxes, (size, size), palette, assignment, K=K, mask_source=mask_source)


def render_clip(
    seq, frames: Seq[int], palette, size: int, *,
    assignment=None, K: int = DEFAULT_K, mask_source: str = "filled_box",
    pseudo_root: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render a clip's (frames, idmaps): two parallel lists of uint8 (size,size,3) arrays.

    Colours are SEQUENCE-STABLE (one fixed colour per track across the clip) so the model
    is supervised to keep each identity's colour through the window. `assignment` defaults
    to the per-sequence stable map; pass an explicit one to share colours across windows of
    the same sequence (needed for cross-window overlap conditioning).
    """
    w, h = seq.image_size
    sx, sy = size / w, size / h
    if assignment is None:
        assignment = sequence_assignment(seq.track_ids, K)
    frames_out, idmaps_out = [], []
    for f in frames:
        frames_out.append(_resize_frame(seq.frame_path(f), size))
        idmaps_out.append(_idmap_at(seq, f, size, palette, assignment, mask_source,
                                    pseudo_root, sx, sy, K))
    return frames_out, idmaps_out


def iter_clips(
    sequences, length: int, size: int, *,
    hop: int | None = None, frame_stride: int = 1, K: int = DEFAULT_K, palette=None,
    mask_source: str = "filled_box", pseudo_root: str | None = None,
    include_occlusion: bool = True, occlusion_strategy: str = "absence",
    min_gap: int = 5, max_gap: int = 40, vis_threshold: float = 0.5,
) -> Iterator[tuple[list[np.ndarray], list[np.ndarray], ClipSpec]]:
    """Yield (frames, idmaps, spec) clips across sequences: sliding windows plus, when
    `include_occlusion`, occlusion-spanning windows (oversampling the hard regime). `hop`
    defaults to `length` (non-overlapping windows)."""
    if palette is None:
        palette = build(K)
    if hop is None:
        hop = length
    for seq in sequences:
        assignment = sequence_assignment(seq.track_ids, K)  # stable across this seq's windows
        specs: Iterator[ClipSpec] = window_specs(seq, length, hop, frame_stride)
        if include_occlusion:
            specs = _chain(specs, occlusion_specs(
                seq, length, frame_stride=frame_stride, strategy=occlusion_strategy,
                min_gap=min_gap, max_gap=max_gap, vis_threshold=vis_threshold))
        for spec in specs:
            frames, idmaps = render_clip(
                seq, spec.frames, palette, size, assignment=assignment, K=K,
                mask_source=mask_source, pseudo_root=pseudo_root)
            yield frames, idmaps, spec


def _chain(*iters: Iterator[ClipSpec]) -> Iterator[ClipSpec]:
    for it in iters:
        yield from it
