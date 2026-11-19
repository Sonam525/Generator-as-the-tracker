"""MOTChallenge format I/O — the SINGLE place tracker outputs are written.

GT line  (gt/gt.txt):  frame, id, bb_left, bb_top, bb_w, bb_h, conf/flag, class, visibility
Pred line (results):   frame, id, bb_left, bb_top, bb_w, bb_h, conf, -1, -1, -1

All values 1-indexed frames, pixel coords (top-left origin), as TrackEval expects.
GT and predictions are kept in the *same* pixel convention so IoU is consistent.
"""
from __future__ import annotations

import configparser
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Box:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0
    cls: int = 1
    vis: float = 1.0

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h


def read_gt(path) -> list[Box]:
    """Parse a MOTChallenge gt.txt / det.txt into Box records.

    Columns past the 6th are optional: conf/flag (7), class (8), visibility (9)
    default to 1.0, 1, 1.0. Blank lines are skipped.
    """
    boxes: list[Box] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        f, tid = int(float(parts[0])), int(float(parts[1]))
        x, y, w, h = (float(parts[i]) for i in range(2, 6))
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        cls = int(float(parts[7])) if len(parts) > 7 else 1
        vis = float(parts[8]) if len(parts) > 8 else 1.0
        boxes.append(Box(f, tid, x, y, w, h, conf, cls, vis))
    return boxes


def write_tracks(boxes: Iterable[Box], path) -> None:
    """Write tracker results in MOTChallenge format (frame-then-id sorted)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(boxes, key=lambda b: (b.frame, b.track_id))
    lines = [
        f"{b.frame},{b.track_id},{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},"
        f"{b.conf:.2f},-1,-1,-1"
        for b in rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def group_by_frame(boxes: Iterable[Box]) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for b in boxes:
        out.setdefault(b.frame, []).append(b)
    return out


def group_by_track(boxes: Iterable[Box]) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for b in boxes:
        out.setdefault(b.track_id, []).append(b)
    for v in out.values():
        v.sort(key=lambda b: b.frame)
    return out


def read_seqinfo(path) -> dict:
    """Parse seqinfo.ini -> dict with imWidth/imHeight/seqLength as ints when present."""
    cp = configparser.ConfigParser()
    cp.read(path)
    sec = cp["Sequence"]
    info: dict = dict(sec)
    for k in ("seqLength", "imWidth", "imHeight", "frameRate"):
        if k in sec:
            info[k] = int(sec[k])
    return info
