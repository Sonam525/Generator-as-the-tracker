"""ID-map sequence -> tracklets by colour.

Pure codec logic: each frame is decoded to {id: mask}; identity across frames is
carried by colour (same colour = same track). No model here. Bbox extraction is
provided for the MOTChallenge writer (data/mot_format.py).
"""
from __future__ import annotations

import numpy as np

from gentrack.color_codec import decode
from gentrack.data.mot_format import Box


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Boolean mask -> (x, y, w, h) in pixels (top-left origin), or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def idmaps_to_boxes(idmaps, palette=None, min_area: int = 0, conf: float = 1.0) -> list[Box]:
    """Canonical ID-map sequence → MOTChallenge Box records (1-indexed frames).

    Each track keeps its palette colour, so decode-free gives {palette_idx: mask};
    the palette index serves as the predicted track id. Bbox = mask bounding box —
    box-IoU against GT boxes is the right metric (silhouette vs filled-box would be
    pessimistic under mask-IoU). Feed the result to mot_format / mot_metrics.
    """
    boxes: list[Box] = []
    for t, rgb in enumerate(idmaps, start=1):
        for pidx, mask in decode(rgb, palette=palette, assignment=None, min_area=min_area).items():
            bb = mask_to_bbox(mask)
            if bb is None:
                continue
            x, y, w, h = bb
            boxes.append(Box(t, int(pidx), float(x), float(y), float(w), float(h), conf=conf))
    return boxes


def decode_sequence(
    idmaps, palette=None, assignment=None, min_area: int = 0
) -> dict[int, dict[int, np.ndarray]]:
    """List of (H,W,3) ID-maps -> {track_id: {frame_idx: mask}}.

    Track id = the decode key (an id when `assignment` is given, else a palette
    index). A track simply collects every frame in which its colour appears.
    """
    tracks: dict[int, dict[int, np.ndarray]] = {}
    for t, rgb in enumerate(idmaps):
        inst = decode(rgb, palette=palette, assignment=assignment, min_area=min_area)
        for tid, mask in inst.items():
            tracks.setdefault(tid, {})[t] = mask
    return tracks
