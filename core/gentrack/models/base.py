"""GeneratorAdapter — the one interface every generator sits behind.

Concrete adapters (cosmos.py, hunyuan.py, nanobanana_api.py) are selected by
config. All methods take and return uint8 RGB arrays of shape (H, W, 3).

Adapters return RAW model output. Cleaning (decode -> correct -> re-encode)
is the pipeline's job (pipeline/track.py), never the adapter's.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GeneratorAdapter(ABC):
    @abstractmethod
    def perceive(self, frame: np.ndarray) -> np.ndarray:
        """Single-frame perception: frame -> raw ID-map (Stage-1 / P0)."""

    @abstractmethod
    def edit(self, frame: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Conditioned generation: (frame, cond) -> raw ID-map.

        `cond` is prepared by the pipeline — an Option-A canvas/composite or an
        Option-B injected-state image. The adapter does not interpret it beyond
        passing it to the model.
        """

    def temporal(self, window: list[np.ndarray]) -> list[np.ndarray]:
        """Option-C video path: a window of frames -> ID-maps. Optional."""
        raise NotImplementedError(f"{type(self).__name__} has no temporal() path")
