"""Option A — identity as pixels: the memory canvas (zero-training baseline, §7.1).

All state lives in the RGB conditioning. For every identity ever seen — including
currently occluded ones — we keep its palette colour, its best appearance crop,
and its recent trajectory. From these we build conditioning images for NB2:

  - paired composite   : previous (RGB, ID-map), alpha-blended (probe rule: never bare maps)
  - memory canvas      : grid of (crop, colour-swatch) tiles for ALL known ids
  - trajectory trails  : recent paths painted as fading colour streaks (motion prior)

State is maintained OUTSIDE the model by the decode->correct->re-encode loop
(pipeline/track.py): feed each frame's cleaned instances to `update`, then call
`conditioning` to assemble the reference images for the next frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from gentrack.palette import DEFAULT_K, build
from gentrack.pipeline.track import make_composite


@dataclass
class _Track:
    color: np.ndarray              # (3,) uint8 palette colour
    best_crop: np.ndarray | None = None
    best_area: int = 0
    centroids: list[tuple[float, float]] = field(default_factory=list)


class MemoryCanvas:
    def __init__(
        self,
        palette: np.ndarray | None = None,
        assignment: dict[int, int] | None = None,
        K: int = DEFAULT_K,
        tile: int = 64,
        max_trail: int = 12,
    ):
        self.palette = palette if palette is not None else build(K)
        self.assignment = assignment
        self.tile = tile
        self.max_trail = max_trail
        self.tracks: dict[int, _Track] = {}

    def _color_for(self, tid: int) -> np.ndarray:
        if self.assignment is not None and tid in self.assignment:
            return self.palette[self.assignment[tid]]
        # fall back to a deterministic slot by insertion order
        idx = (len(self.tracks) % (len(self.palette) - 1)) + 1
        return self.palette[idx]

    def update(self, frame: np.ndarray, instances: dict[int, np.ndarray]) -> None:
        """Fold one frame's instances ({id: bool mask}) into the canvas state."""
        for tid, mask in instances.items():
            area = int(mask.sum())
            if area == 0:
                continue
            tr = self.tracks.get(tid)
            if tr is None:
                tr = _Track(color=np.asarray(self._color_for(tid), dtype=np.uint8))
                self.tracks[tid] = tr

            ys, xs = np.nonzero(mask)
            tr.centroids.append((float(xs.mean()), float(ys.mean())))
            if len(tr.centroids) > self.max_trail:
                tr.centroids.pop(0)

            if area > tr.best_area:
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
                crop = frame[y0 : y1 + 1, x0 : x1 + 1].copy()
                crop[~mask[y0 : y1 + 1, x0 : x1 + 1]] = 0  # black out background
                tr.best_crop = crop
                tr.best_area = area

    def _tile_image(self, tr: _Track) -> np.ndarray:
        t = self.tile
        swatch_h = max(6, t // 4)
        out = np.zeros((t, t, 3), dtype=np.uint8)
        if tr.best_crop is not None and tr.best_crop.size:
            crop_h = t - swatch_h
            resized = Image.fromarray(tr.best_crop, mode="RGB").resize(
                (t, crop_h), Image.NEAREST
            )
            out[:crop_h] = np.asarray(resized, dtype=np.uint8)
        out[t - swatch_h :] = tr.color  # colour swatch strip at the bottom
        return out

    def build_canvas(self) -> np.ndarray:
        """Grid of (crop, swatch) tiles for all known ids (incl. occluded)."""
        ids = sorted(self.tracks)
        t = self.tile
        if not ids:
            return np.zeros((t, t, 3), dtype=np.uint8)
        cols = int(np.ceil(np.sqrt(len(ids))))
        rows = int(np.ceil(len(ids) / cols))
        canvas = np.zeros((rows * t, cols * t, 3), dtype=np.uint8)
        for k, tid in enumerate(ids):
            r, c = divmod(k, cols)
            canvas[r * t : (r + 1) * t, c * t : (c + 1) * t] = self._tile_image(self.tracks[tid])
        return canvas

    def render_trails(self, size: tuple[int, int], dot: int = 3) -> np.ndarray:
        """Fading colour trails per id on a black canvas. `size` = (width, height)."""
        w, h = size
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for tr in self.tracks.values():
            n = len(tr.centroids)
            for k, (cx, cy) in enumerate(tr.centroids):
                fade = (k + 1) / n  # newest brightest
                col = (tr.color.astype(np.float32) * fade).astype(np.uint8)
                x, y = int(round(cx)), int(round(cy))
                x0, x1 = max(0, x - dot), min(w, x + dot + 1)
                y0, y1 = max(0, y - dot), min(h, y + dot + 1)
                out[y0:y1, x0:x1] = col
        return out

    def conditioning(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        prev_idmap: np.ndarray,
        *,
        with_trails: bool = True,
    ) -> list[np.ndarray]:
        """Assemble the NB2 reference list (<=14): the PREVIOUS (frame, ID-map)
        paired composite, the memory canvas, and trails. `frame` is the target the
        adapter segments (passed separately by the adapter), used here only for size.
        """
        refs = [make_composite(prev_frame, prev_idmap), self.build_canvas()]
        if with_trails:
            refs.append(self.render_trails((frame.shape[1], frame.shape[0])))
        return refs
