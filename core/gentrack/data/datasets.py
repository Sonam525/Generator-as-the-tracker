"""DanceTrack / MOT17 ingestion over the MOTChallenge directory layout.

    <root>/<split>/<seq>/img1/*.{jpg,png}
    <root>/<split>/<seq>/gt/gt.txt
    <root>/<split>/<seq>/seqinfo.ini   (optional; size inferred from a frame if absent)

GT boxes+ids are the identity labels for both datasets (box-only — masks come
later from data/pseudo_masks.py). MOT17 ships -FRCNN/-DPM/-SDP detector variants
with shared GT; each variant dir is loaded as its own Sequence — the caller picks.

NOTE (flagged, not silently decided): the MOT17 train/val "half" protocol is
ambiguous in the plan (§8.1 reads as a sequence split; "train-half val" usually
means a per-sequence frame split). Not baked in here — to be pinned when MOT17
eval is wired. DanceTrack uses its own train/val/test dirs directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from gentrack.data.mot_format import Box, group_by_frame, read_gt, read_seqinfo

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sequence:
    name: str
    root: Path
    frames: list[Path]              # sorted image paths (frame i -> frames[i-1])
    gt: list[Box]
    info: dict
    _by_frame: dict[int, list[Box]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._by_frame = group_by_frame(self.gt)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height) from seqinfo, else inferred from the first frame."""
        if "imWidth" in self.info and "imHeight" in self.info:
            return int(self.info["imWidth"]), int(self.info["imHeight"])
        with Image.open(self.frames[0]) as im:
            return im.size

    def boxes_at(self, frame: int) -> list[Box]:
        """GT boxes at a 1-indexed frame number."""
        return self._by_frame.get(frame, [])

    def frame_path(self, frame: int) -> Path:
        return self.frames[frame - 1]

    @property
    def track_ids(self) -> list[int]:
        return sorted({b.track_id for b in self.gt})


def _list_frames(img_dir: Path) -> list[Path]:
    # Skip hidden / AppleDouble sidecars ("._000001.jpg" ship inside the SportsMOT tars): they are
    # not images and would corrupt the frame-index -> file mapping.
    files = [p for p in img_dir.iterdir()
             if p.suffix.lower() in _IMG_EXTS and not p.name.startswith((".", "._"))]
    return sorted(files, key=lambda p: p.name)


def load_sequence(seq_dir) -> Sequence:
    seq_dir = Path(seq_dir)
    img_dir = seq_dir / "img1"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"no img1/ under {seq_dir}")
    frames = _list_frames(img_dir)
    if not frames:
        raise FileNotFoundError(f"no images in {img_dir}")

    gt_path = seq_dir / "gt" / "gt.txt"
    gt = read_gt(gt_path) if gt_path.exists() else []

    info_path = seq_dir / "seqinfo.ini"
    info = read_seqinfo(info_path) if info_path.exists() else {"name": seq_dir.name}
    return Sequence(name=seq_dir.name, root=seq_dir, frames=frames, gt=gt, info=info)


def dancetrack_splits(root, train_dirs, internal_val, test_dirs):
    """Return (train, val, test) Sequence lists per the P0+ protocol (datasets/data.yaml):
    train = `train_dirs` videos MINUS `internal_val`; val = the `internal_val` videos
    (model selection); test = `test_dirs` (the 25 official val, used as test). The
    split is video-level — no frames from one video land in two splits.
    """
    held = set(internal_val)
    all_train = load_split(root, train_dirs)
    train = [s for s in all_train if s.name not in held]
    val = [s for s in all_train if s.name in held]
    test = load_split(root, test_dirs)
    return train, val, test


def _subseq(seq: Sequence, start: int, end: int) -> Sequence:
    """A self-contained Sequence over original frames [start, end] (1-indexed, inclusive),
    re-indexed so its own frames run 1..(end-start+1). Track ids and per-box fields are
    preserved; frame numbers are offset by (start-1)."""
    off = start - 1
    frames = seq.frames[off:end]
    gt = [Box(b.frame - off, b.track_id, b.x, b.y, b.w, b.h, b.conf, b.cls, b.vis)
          for b in seq.gt if start <= b.frame <= end]
    return Sequence(name=f"{seq.name}_f{start}-{end}", root=seq.root, frames=frames, gt=gt, info=dict(seq.info))


def frame_half(seq: Sequence) -> tuple[Sequence, Sequence]:
    """Split a Sequence into (train_half, val_half) = first 50% / last 50% of its frames,
    each a self-contained re-indexed Sequence (standard MOT17 half-split ablation protocol).
    Keeps every scene in both halves; val frames are unseen."""
    n = seq.num_frames
    mid = n // 2
    return _subseq(seq, 1, mid), _subseq(seq, mid + 1, n)


def mot17_splits(root, split_dir: str = "train") -> tuple[list[Sequence], list[Sequence]]:
    """Return (train_halves, val_halves) for MOT17: load every sequence under
    <root>/<split_dir>/ (one detector variant on disk) and frame-half each. Per-sequence
    frame split → many val occlusion events for re-acquisition while scenes stay in both."""
    train, val = [], []
    for s in load_split(root, split_dir):
        th, vh = frame_half(s)
        train.append(th)
        val.append(vh)
    return train, val


def load_split(root, split) -> list[Sequence]:
    """Load every sequence under <root>/<split>/, sorted by sequence name.

    `split` is a dir name or a list of dir names. DanceTrack's HF distribution
    splits the train set across `train1/` and `train2/` (and test across
    `test1/`/`test2/`), so pass e.g. ["train1", "train2"] to get the full train
    set as one list — no slow file-moves on blob storage required.
    """
    names = [split] if isinstance(split, str) else list(split)
    seqs: list[Sequence] = []
    for name in names:
        split_dir = Path(root) / name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split dir not found: {split_dir}")
        seq_dirs = sorted(p for p in split_dir.iterdir() if (p / "img1").is_dir())
        seqs.extend(load_sequence(p) for p in seq_dirs)
    seqs.sort(key=lambda s: s.name)
    return seqs
