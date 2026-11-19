"""Build ID-maps and training tuples from GT (§8.3).

- Stage-1 pairs:        (frame -> ID-map)
- Stage-2 temporal:     (frame_t, ID-map_t, frame_{t+stride}) -> ID-map_{t+stride}
- Occlusion triples:    mined visible -> occluded -> reappears spans, oversampled

Mask source is a flag (filled_box | sam_pseudo); only filled_box is implemented
here — SAM pseudo-masks come from data/pseudo_masks.py. Parameters are pinned in
datasets/data.yaml (palette K, mask_source, paint order, mining gaps/threshold).

Overlap/paint order: boxes are painted low-visibility-first so the most-visible
(front) instance ends up on top — the best filled-box proxy for depth ordering.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence as Seq
from dataclasses import dataclass

import numpy as np

from gentrack.color_codec import encode
from gentrack.data.datasets import Sequence
from gentrack.data.mot_format import Box, group_by_track
from gentrack.palette import DEFAULT_K, assign, build


def sequence_assignment(track_ids: Seq[int], K: int = DEFAULT_K) -> dict[int, int]:
    """Stable per-sequence id->colour map. Raises if a sequence has > K-1 ids.

    Works for DanceTrack (few ids/seq). MOT17 sequences routinely exceed K-1
    unique tracks and will need colour *recycling* (reuse a freed colour once a
    track is long gone) — that's an additive change to palette.py and needs human
    sign-off (frozen zone), so it is deliberately NOT done here yet.
    """
    return assign(track_ids, K)


def _box_to_slice(b: Box, w: int, h: int):
    x0 = max(0, int(round(b.x)))
    y0 = max(0, int(round(b.y)))
    x1 = min(w, int(round(b.x + b.w)))
    y1 = min(h, int(round(b.y + b.h)))
    return y0, y1, x0, x1


def filled_box_masks(boxes: Seq[Box], size: tuple[int, int]):
    """(masks, ids) from boxes, painted low-vis first (front instance on top).

    `size` is (width, height). Degenerate/out-of-frame boxes are skipped.
    """
    w, h = size
    ordered = sorted(boxes, key=lambda b: (b.vis, b.track_id))
    masks, ids = [], []
    for b in ordered:
        y0, y1, x0, x1 = _box_to_slice(b, w, h)
        if y1 <= y0 or x1 <= x0:
            continue
        m = np.zeros((h, w), dtype=bool)
        m[y0:y1, x0:x1] = True
        masks.append(m)
        ids.append(b.track_id)
    return masks, ids


def render_idmap(
    boxes: Seq[Box],
    size: tuple[int, int],
    palette=None,
    assignment=None,
    K: int = DEFAULT_K,
    mask_source: str = "filled_box",
) -> np.ndarray:
    """Render a single frame's GT ID-map. `size` = (width, height)."""
    if mask_source == "sam_pseudo":
        raise NotImplementedError("SAM pseudo-masks come from data/pseudo_masks.py")
    if mask_source != "filled_box":
        raise ValueError(f"unknown mask_source {mask_source!r}")

    w, h = size
    if palette is None:
        palette = build(K)
    masks, ids = filled_box_masks(boxes, size)
    if not masks:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if assignment is None:
        assignment = assign(ids, K)
    return encode(masks, ids, palette=palette, assignment=assignment)


def stage1_pairs(
    seq: Sequence, palette=None, assignment=None, mask_source: str = "filled_box"
) -> Iterator[tuple[int, "object", np.ndarray]]:
    """Yield (frame_idx, frame_path, idmap) for every frame in the sequence."""
    size = seq.image_size
    if palette is None:
        palette = build()
    if assignment is None:
        assignment = sequence_assignment(seq.track_ids)
    for f in range(1, seq.num_frames + 1):
        idmap = render_idmap(seq.boxes_at(f), size, palette, assignment,
                             mask_source=mask_source)
        yield f, seq.frame_path(f), idmap


@dataclass
class TemporalTriple:
    frame_t: int
    path_t: object
    idmap_t: np.ndarray
    frame_tk: int
    path_tk: object
    idmap_tk: np.ndarray
    stride: int


def temporal_triples(
    seq: Sequence, stride: int = 1, palette=None, assignment=None,
    mask_source: str = "filled_box",
) -> Iterator[TemporalTriple]:
    """Yield temporal triples at the given stride (§8.3 Stage-2)."""
    size = seq.image_size
    if palette is None:
        palette = build()
    if assignment is None:
        assignment = sequence_assignment(seq.track_ids)

    def render(f):
        return render_idmap(seq.boxes_at(f), size, palette, assignment,
                            mask_source=mask_source)

    for f in range(1, seq.num_frames - stride + 1):
        yield TemporalTriple(
            f, seq.frame_path(f), render(f),
            f + stride, seq.frame_path(f + stride), render(f + stride),
            stride,
        )


@dataclass
class OcclusionEvent:
    track_id: int
    before_frame: int   # last visible frame before the occlusion (1-indexed)
    after_frame: int    # first visible frame after it
    gap: int            # length of the occlusion (frames)
    kind: str           # "absence" (DanceTrack) | "low_visibility" (MOT17)


def mine_absence(boxes: Seq[Box], min_gap: int = 5, max_gap: int = 40) -> list[OcclusionEvent]:
    """DanceTrack strategy: a track disappears from gt.txt then reappears."""
    events: list[OcclusionEvent] = []
    for tid, track in group_by_track(boxes).items():
        frames = [b.frame for b in track]
        for a, c in zip(frames, frames[1:]):
            gap = c - a - 1
            if min_gap <= gap <= max_gap:
                events.append(OcclusionEvent(tid, a, c, gap, "absence"))
    return events


def mine_low_visibility(
    boxes: Seq[Box], vis_threshold: float = 0.5, min_gap: int = 5, max_gap: int = 40
) -> list[OcclusionEvent]:
    """MOT17 strategy: a run of low-visibility frames flanked by visible ones."""
    events: list[OcclusionEvent] = []
    for tid, track in group_by_track(boxes).items():
        n = len(track)
        i = 0
        while i < n:
            if track[i].vis < vis_threshold:
                j = i
                while j + 1 < n and track[j + 1].vis < vis_threshold:
                    j += 1
                run = j - i + 1
                before = track[i - 1].frame if i > 0 else None
                after = track[j + 1].frame if j + 1 < n else None
                if before is not None and after is not None and min_gap <= run <= max_gap:
                    events.append(OcclusionEvent(tid, before, after, run, "low_visibility"))
                i = j + 1
            else:
                i += 1
    return events


def mine_occlusion(
    boxes: Seq[Box], strategy: str = "absence", *,
    min_gap: int = 5, max_gap: int = 40, vis_threshold: float = 0.5,
) -> list[OcclusionEvent]:
    """Dispatch to the dataset-appropriate mining strategy (§8.3)."""
    if strategy == "absence":
        return mine_absence(boxes, min_gap, max_gap)
    if strategy == "low_visibility":
        return mine_low_visibility(boxes, vis_threshold, min_gap, max_gap)
    raise ValueError(f"unknown strategy {strategy!r}")
