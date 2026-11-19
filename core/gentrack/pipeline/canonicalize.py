"""Canonical tracker — the decode→match→canonicalise→re-encode correction (§11).

NB2 keeps each identity's colour only *perceptually* (it quantises to neighbouring
palette indices frame to frame; finding gentrack-nb2-findings), so we cannot decode
against a fixed assignment. Instead, each frame:

  1. decode the raw map FREE → blobs, each tagged with the palette index its colour
     landed on;
  2. match blobs to existing tracks by a cost of (1 - IoU) + colour distance, via
     Hungarian assignment, gated so only spatially-or-perceptually-close pairs match;
  3. matched blobs keep their track's FIXED canonical colour; unmatched blobs start
     new tracks; unmatched tracks age (kept `max_missing` frames for re-acquisition).

`update` returns the visible {canonical_id: mask}; `assignment()` gives
canonical_id → palette index for re-encoding. This is what makes the chained
conditioning use one stable colour per identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from gentrack.color_codec import decode
from gentrack.palette import DEFAULT_K, _rgb_to_lab

_BIG = 1e6
_COLOR_SCALE = 50.0  # normalises a CIELAB ΔE into ~[0,1] for the cost


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter / union)


@dataclass
class _Track:
    palette_idx: int          # fixed canonical colour for this identity
    last_mask: np.ndarray
    last_color_idx: int       # NB2's palette index last seen (for colour matching)
    missing: int = 0


class CanonicalTracker:
    def __init__(
        self,
        palette: np.ndarray,
        K: int = DEFAULT_K,
        min_area: int = 200,
        w_color: float = 1.0,
        min_iou: float = 0.05,
        max_color: float = 25.0,
        max_missing: int = 50,
    ):
        self.palette = palette
        self.K = K
        self.min_area = min_area
        self.w_color = w_color
        self.min_iou = min_iou
        self.max_color = max_color
        self.max_missing = max_missing
        self._lab = _rgb_to_lab(palette.astype(np.float64) / 255.0)  # (K, 3)
        self.tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._used: set[int] = set()

    def assignment(self) -> dict[int, int]:
        return {tid: t.palette_idx for tid, t in self.tracks.items()}

    def _free_palette_idx(self) -> int:
        for i in range(1, len(self.palette)):
            if i not in self._used:
                return i
        raise RuntimeError("palette exhausted: >K-1 simultaneous tracks (need recycling)")

    def _color_dist(self, i: int, j: int) -> float:
        return float(np.sqrt(np.sum((self._lab[i] - self._lab[j]) ** 2)))

    def update(self, raw: np.ndarray, frame=None) -> dict[int, np.ndarray]:
        dets = list(
            decode(raw, palette=self.palette, assignment=None, min_area=self.min_area).items()
        )
        tids = list(self.tracks)

        pairs: list[tuple[int, int]] = []
        matched_det: set[int] = set()
        matched_trk: set[int] = set()
        if dets and tids:
            cost = np.full((len(dets), len(tids)), _BIG)
            feasible = np.zeros_like(cost, dtype=bool)
            for di, (cidx, mask) in enumerate(dets):
                for ti, tid in enumerate(tids):
                    t = self.tracks[tid]
                    iou = _iou(mask, t.last_mask)
                    cdist = self._color_dist(cidx, t.last_color_idx)
                    if iou >= self.min_iou or cdist <= self.max_color:
                        cost[di, ti] = (1.0 - iou) + self.w_color * (cdist / _COLOR_SCALE)
                        feasible[di, ti] = True
            rows, cols = linear_sum_assignment(cost)
            for di, ti in zip(rows, cols):
                if feasible[di, ti]:
                    pairs.append((di, ti))
                    matched_det.add(di)
                    matched_trk.add(ti)

        visible: dict[int, np.ndarray] = {}
        for di, ti in pairs:
            cidx, mask = dets[di]
            t = self.tracks[tids[ti]]
            t.last_mask, t.last_color_idx, t.missing = mask, cidx, 0
            visible[tids[ti]] = mask

        for di, (cidx, mask) in enumerate(dets):
            if di in matched_det:
                continue
            pidx = self._free_palette_idx()
            self._used.add(pidx)
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = _Track(palette_idx=pidx, last_mask=mask, last_color_idx=cidx)
            visible[tid] = mask

        for ti, tid in enumerate(tids):
            if ti in matched_trk:
                continue
            t = self.tracks[tid]
            t.missing += 1
            if t.missing > self.max_missing:
                self._used.discard(t.palette_idx)
                del self.tracks[tid]

        return visible
