"""Chained tracking eval for trained generators (image-mode, composite conditioning).

Autoregress a generator over a clip exactly as it was trained (P1/P2): each frame is
conditioned on a composite of (current frame, previous CLEANED ID-map); the raw output
is decoded-free, matched to canonical tracks (CanonicalTracker), and re-encoded — the
cleaned map (never raw output) conditions the next frame (decode→correct→re-encode).

Torch-free: `adapter` only needs `.generate(cond_rgb) -> idmap_rgb` (both uint8 HWC), so
this is unit-testable with a mock and works for any trained Cosmos-Nano LoRA. The result
(canonical clean maps) feeds idmaps_to_boxes → MOT identity metrics + reacquisition.
"""
from __future__ import annotations

import numpy as np

from gentrack.palette import DEFAULT_K, build
from gentrack.pipeline.canonicalize import CanonicalTracker
from gentrack.pipeline.track import make_composite, reencode


def chained_track(
    adapter,
    frames,
    *,
    palette=None,
    K: int = DEFAULT_K,
    min_area: int = 300,
    alpha: float = 0.5,
) -> list[np.ndarray]:
    """Run the generator autoregressively over `frames`; return one canonical clean
    ID-map per frame. Frame 0 conditions on a black previous map (composite = dimmed
    frame), matching how the model saw an empty history at training start.
    """
    if palette is None:
        palette = build(K)
    tracker = CanonicalTracker(palette, K=K, min_area=min_area)
    clean_maps: list[np.ndarray] = []
    prev_clean: np.ndarray | None = None
    for frame in frames:
        prev = prev_clean if prev_clean is not None else np.zeros_like(frame)
        raw = adapter.generate(make_composite(frame, prev, alpha=alpha))
        inst = tracker.update(raw, frame)
        clean = reencode(inst, frame.shape[:2], palette, tracker.assignment())
        clean_maps.append(clean)
        prev_clean = clean
    return clean_maps


def perframe_track(
    adapter,
    frames,
    *,
    palette=None,
    K: int = DEFAULT_K,
    min_area: int = 300,
) -> list[np.ndarray]:
    """No-state floor: condition each frame on the frame ALONE — there is no pixel
    state to copy, so re-perception is forced. The CanonicalTracker threads identities
    across frames by IoU+colour, so this is the fair no-state control (per-frame
    perception + heuristic association, no learned memory) — the Gate-2B baseline P2's
    bank must beat.

    The trained Cosmos-Nano adapter conditions on the raw frame via i2v spatial packing
    (frame = clean condition latent), so we pass the frame directly. (Pre-i2v this fed a
    dimmed composite; the conditioning fix made that obsolete.)
    """
    if palette is None:
        palette = build(K)
    tracker = CanonicalTracker(palette, K=K, min_area=min_area)
    clean_maps: list[np.ndarray] = []
    for frame in frames:
        raw = adapter.generate(frame)
        inst = tracker.update(raw, frame)
        clean = reencode(inst, frame.shape[:2], palette, tracker.assignment())
        clean_maps.append(clean)
    return clean_maps
