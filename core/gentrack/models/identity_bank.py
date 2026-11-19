"""Option B — the per-track identity bank (the headline contribution; §7.2).

Holds, per identity (= palette colour), an **appearance embedding** that is momentum-
updated while the track is visible and **frozen while it is occluded**, plus a motion
state (last box + velocity) that is extrapolated under occlusion. On reappearance the
generator re-binds the identity by being conditioned on the still-present embedding —
re-acquisition as subject recognition rather than a tracking heuristic.

This module is ONLY the state (where identity lives + how it is maintained), decoupled
from the backbone and the injection mechanism so it is unit-testable without a GPU:
embeddings are arbitrary-dim float vectors (the appearance encoder decides the dim) and
nothing here imports torch. The flow each frame is:

    cond = bank.query()                      # (embedding, colour) for every live track
    idmap = generator(frame, cond)           # injection is the model's job (Stage-3)
    clean = decode -> correct -> re-encode    # rule #4
    bank.update(frame_idx, observations)     # momentum visible / freeze occluded

`observations` is read off the *cleaned* map (colour = identity), so association is done
by the generator + codec, not here. Identity is never invented in this module.

Experimental parameters (momentum, max_misses) are UNPINNED defaults — promote to
`image_generators/cosmos/train.yaml` before any Stage-3 run (parameters are pinned before a run; README.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]  # (x, y, w, h)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    color: int                  # palette index = stable identity
    embedding: np.ndarray       # unit-norm appearance vector
    box: Box                    # last visible box, or motion-extrapolated while occluded
    velocity: tuple[float, float]
    last_seen: int
    age: int = 1                # visible-frame count
    misses: int = 0             # consecutive occluded frames

    @property
    def occluded(self) -> bool:
        return self.misses > 0


class IdentityBank:
    def __init__(self, *, momentum: float = 0.9, max_misses: int = 50):
        # momentum: EMA weight on the OLD embedding when visible (higher = slower drift).
        # max_misses: drop a track after this many consecutive occluded frames.
        self.momentum = momentum
        self.max_misses = max_misses
        self.tracks: dict[int, Track] = {}   # colour -> Track

    def update(self, frame_idx: int, observations: dict[int, tuple[np.ndarray, Box]]) -> None:
        """Momentum-update embeddings of visible tracks; freeze + motion-extrapolate the
        occluded ones. `observations` maps colour -> (appearance embedding, box) for the
        identities present in this frame's cleaned ID-map."""
        seen = set(observations)
        for color, (emb, box) in observations.items():
            emb = _unit(emb)
            tr = self.tracks.get(color)
            if tr is None:
                self.tracks[color] = Track(color, emb, box, (0.0, 0.0), frame_idx)
                continue
            gap = max(1, frame_idx - tr.last_seen)
            tr.velocity = ((box[0] - tr.box[0]) / gap, (box[1] - tr.box[1]) / gap)
            tr.embedding = _unit(self.momentum * tr.embedding + (1.0 - self.momentum) * emb)
            tr.box = box
            tr.last_seen = frame_idx
            tr.age += 1
            tr.misses = 0
        for color in list(self.tracks):
            if color in seen:
                continue
            tr = self.tracks[color]
            tr.misses += 1
            # embedding is FROZEN (not touched); only the predicted box advances.
            tr.box = (tr.box[0] + tr.velocity[0], tr.box[1] + tr.velocity[1], tr.box[2], tr.box[3])
            if tr.misses > self.max_misses:
                del self.tracks[color]

    def assign(self, detections, frame_idx: int, *, w_app: float = 0.5, w_motion: float = 0.5,
               match_thr: float = 0.6, K: int = 48) -> list[int | None]:
        """Option-3 re-id: match per-frame detections to bank tracks by appearance (cosine)
        + motion (predicted-box IoU), start new tracks for the unmatched, then update the
        bank (momentum visible / freeze occluded). Returns the assigned palette colour per
        detection (parallel to `detections`), or None if no colour was free.

        `detections` is a list of (appearance embedding, box). This is where identity lives
        in the re-id pivot — the generator only detects; association happens here against the
        persistent (frozen-under-occlusion) embeddings, so a reappearing track re-binds by
        appearance+motion even when its IoU with the last visible frame is 0.

        UNPINNED params (propose; pin in image_generators/cosmos/train.yaml before a real run): w_app,
        w_motion, match_thr. On DanceTrack appearance is ambiguous → motion likely weights
        higher; tune on internal-val, never on the metric clip.
        """
        n = len(detections)
        out: list[int | None] = [None] * n
        track_colors = list(self.tracks)
        if track_colors and n:
            cost = np.full((n, len(track_colors)), 1e9, dtype=np.float32)
            for i, (emb, box) in enumerate(detections):
                e = _unit(emb)
                for j, c in enumerate(track_colors):
                    tr = self.tracks[c]
                    app = 1.0 - float(np.dot(e, tr.embedding))     # both unit-norm
                    mot = 1.0 - _iou(box, tr.box)                  # tr.box = predicted under occlusion
                    cost[i, j] = w_app * app + w_motion * mot
            used_i: set[int] = set()
            used_j: set[int] = set()
            for _c, i, j in sorted((cost[i, j], i, j) for i in range(n) for j in range(len(track_colors))):
                if _c > match_thr or i in used_i or j in used_j:
                    continue
                out[i] = track_colors[j]
                used_i.add(i)
                used_j.add(j)
        # unmatched detections -> new tracks (allocate a free palette colour, 1..K-1)
        used = {c for c in out if c is not None} | set(self.tracks)
        free = [c for c in range(1, K) if c not in used]
        fi = 0
        obs: dict[int, tuple[np.ndarray, Box]] = {}
        for i, (emb, box) in enumerate(detections):
            if out[i] is None:
                if fi >= len(free):
                    continue                                       # palette exhausted; drop
                out[i] = free[fi]
                fi += 1
            obs[out[i]] = (emb, box)
        self.update(frame_idx, obs)
        return out

    def query(self) -> list[tuple[np.ndarray, int]]:
        """(embedding, colour) for every live track — including occluded ones, which is
        the whole point: their frozen embedding is what lets the generator re-bind them."""
        return [(tr.embedding, tr.color) for tr in self.tracks.values()]

    def predicted_box(self, color: int) -> Box | None:
        tr = self.tracks.get(color)
        return tr.box if tr is not None else None

    def colors(self) -> set[int]:
        return set(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)
