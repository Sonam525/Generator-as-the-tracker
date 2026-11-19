"""Chained tracking: the decode -> correct -> re-encode loop (rule 4 of the working agreement, README.md).

Every generated map is decoded to instances, sanity-corrected against the
previous state, and *re-rendered* through the codec. The cleaned map — never the
raw model output — is what conditions the next frame and writes downstream state
(memory canvas / embedding bank). This is the standard drift mitigation for
chained generation.

What's here is the mechanism. The conditioning builder (`make_composite`) is the
zero-state Option-A starting point; richer conditioning (memory canvas, trails)
lives in pipeline/memory_canvas.py, and motion/IoU re-acquisition correction in
pipeline/reacquire.py. Those plug into `condition_fn` / `correct_fn` below.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from gentrack.color_codec import decode, encode
from gentrack.models.base import GeneratorAdapter
from gentrack.palette import DEFAULT_K, build


def make_composite(frame: np.ndarray, idmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Alpha-blend an RGB frame with its paired ID-map (probe rule: never bare maps).

    Keeps (H,W,3) shape so adapters need no special handling. Side-by-side is the
    alternative the probe also sanctioned; swap in via `condition_fn` if preferred.
    """
    blend = (1.0 - alpha) * frame.astype(np.float32) + alpha * idmap.astype(np.float32)
    return np.clip(blend, 0, 255).astype(np.uint8)


def reencode(inst: dict[int, np.ndarray], shape, palette, assignment) -> np.ndarray:
    """{id: mask} -> clean ID-map. Empty -> all-black map of `shape` (H,W)."""
    h, w = shape
    if not inst:
        return np.zeros((h, w, 3), dtype=np.uint8)
    ids = list(inst.keys())
    masks = [inst[i] for i in ids]
    return encode(masks, ids, palette=palette, assignment=assignment)


def _default_correct(inst, prev_inst):
    """Pass-through correction (extension point).

    The re-encode step already removes the dominant drift source (off-palette /
    gradient / outline leakage) by routing raw output through the codec. Motion /
    IoU continuity and re-acquisition land in reacquire.py at P3.
    """
    return inst


def run_chain(
    adapter: GeneratorAdapter,
    frames: Sequence[np.ndarray],
    *,
    init_idmap: np.ndarray | None = None,
    K: int = DEFAULT_K,
    palette: np.ndarray | None = None,
    assignment: dict[int, int] | None = None,
    min_area: int = 4,
    condition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = make_composite,
    correct_fn: Callable[[dict, dict], dict] = _default_correct,
    state=None,
    tracker=None,
    clean_loop: bool = True,
) -> list[np.ndarray]:
    """Run chained tracking over `frames`; return one cleaned ID-map per frame.

    First frame: `adapter.perceive` unless `init_idmap` is given. Subsequent
    frames: `adapter.edit(frame, cond)`. Each raw output is decoded, corrected,
    and re-encoded; the returned maps are always the cleaned ones.

    `state` (e.g. an Option-A MemoryCanvas / Option-B bank) is the conditioning +
    write-path hook: when given, `cond = state.conditioning(frame, prev_frame,
    prev_cond)` and `state.update(frame, inst)` is called each frame; otherwise
    `condition_fn(prev_frame, prev_cond)` is used. The reference is always the
    PREVIOUS (frame, ID-map) pair (probe rule §4) — `frame` is the target the
    adapter segments. `clean_loop=False` ablates the re-encode loop — the RAW model
    output (not the cleaned map) conditions the next frame, while metrics still use
    the cleaned maps. Always decode→correct→re-encode (rule #4) unless ablating.

    `tracker` (a CanonicalTracker) replaces the fixed-assignment decode with
    decode-free + IoU/colour matching to stable canonical colours — required for
    real NB2 output, which doesn't reproduce exact palette RGB.
    """
    if palette is None:
        palette = build(K)

    clean_maps: list[np.ndarray] = []
    prev_cond = init_idmap  # conditions the next frame (cleaned map, or raw if ablating)
    prev_frame: np.ndarray | None = None
    prev_inst = (
        decode(prev_cond, palette=palette, assignment=assignment, min_area=min_area)
        if prev_cond is not None
        else {}
    )

    for frame in frames:
        if prev_cond is None or prev_frame is None:
            raw = adapter.perceive(frame)
        else:
            cond = (
                state.conditioning(frame, prev_frame, prev_cond)
                if state is not None
                else condition_fn(prev_frame, prev_cond)
            )
            raw = adapter.edit(frame, cond)

        if tracker is not None:
            # decode-free → match to canonical tracks → re-encode (canonicalize.py)
            inst = tracker.update(raw, frame)
            cur_assignment = tracker.assignment()
        else:
            inst = decode(raw, palette=palette, assignment=assignment, min_area=min_area)
            inst = correct_fn(inst, prev_inst)
            cur_assignment = assignment
        clean = reencode(inst, frame.shape[:2], palette, cur_assignment)

        clean_maps.append(clean)
        if state is not None:
            state.update(frame, inst)
        prev_cond = clean if clean_loop else raw
        prev_inst = inst
        prev_frame = frame

    return clean_maps
