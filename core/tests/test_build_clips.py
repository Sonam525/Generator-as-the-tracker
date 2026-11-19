"""Clip builder: windowing math, clip shapes, sequence-stable colours, occlusion windows."""
import numpy as np
from PIL import Image

from gentrack.color_codec import decode
from gentrack.data.build_clips import (
    clip_frames,
    clip_starts,
    iter_clips,
    occlusion_specs,
    render_clip,
    window_specs,
)
from gentrack.data.build_idmaps import sequence_assignment
from gentrack.data.datasets import load_sequence
from gentrack.palette import build


def _make_seq(seq_dir, n=6, w=160, h=80):
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8)).save(img_dir / f"{i:06d}.png")
    gt = seq_dir / "gt"
    gt.mkdir()
    lines = []
    for f in range(1, n + 1):
        lines.append(f"{f},1,{10 + f},10,40,50,1,1,1")   # id 1 drifts right
        lines.append(f"{f},2,100,10,30,50,1,1,1")         # id 2 static
    (gt / "gt.txt").write_text("\n".join(lines) + "\n")
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nseqLength={n}\n"
        f"imWidth={w}\nimHeight={h}\nimExt=.png\n"
    )


def _make_seq_with_gap(seq_dir, n=20, w=160, h=80):
    """id 1 visible f1-5 and f12-20 (gap 6-11 => gap len 6); id 2 visible throughout."""
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8)).save(img_dir / f"{i:06d}.png")
    gt = seq_dir / "gt"
    gt.mkdir()
    lines = []
    for f in range(1, n + 1):
        if f <= 5 or f >= 12:
            lines.append(f"{f},1,{10 + f},10,40,50,1,1,1")
        lines.append(f"{f},2,100,10,30,50,1,1,1")
    (gt / "gt.txt").write_text("\n".join(lines) + "\n")
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nseqLength={n}\n"
        f"imWidth={w}\nimHeight={h}\nimExt=.png\n"
    )


def test_clip_starts_and_frames():
    assert clip_starts(10, 4, hop=4) == [1, 5]            # last fit = 7 -> [1,5]
    assert clip_starts(10, 4, hop=2) == [1, 3, 5, 7]
    assert clip_starts(10, 3, hop=1, frame_stride=2) == [1, 2, 3, 4, 5, 6]  # cover 5 -> last 6
    assert clip_starts(4, 8, hop=8) == []                 # window does not fit
    assert clip_frames(3, 4) == [3, 4, 5, 6]
    assert clip_frames(1, 3, frame_stride=2) == [1, 3, 5]


def test_window_specs_cover_sequence(tmp_path):
    seq_dir = tmp_path / "dancetrack0001"
    _make_seq(seq_dir, n=6)
    seq = load_sequence(seq_dir)
    specs = list(window_specs(seq, length=4, hop=2))
    assert [s.frames for s in specs] == [[1, 2, 3, 4], [3, 4, 5, 6]]
    assert all(s.kind == "window" for s in specs)


def test_render_clip_shapes_and_sequence_stable_colours(tmp_path):
    seq_dir = tmp_path / "dancetrack0001"
    _make_seq(seq_dir, n=6)
    seq = load_sequence(seq_dir)
    pal = build(48)
    a = sequence_assignment(seq.track_ids, 48)
    frames, idmaps = render_clip(seq, [1, 2, 3, 4], pal, size=128)
    assert len(frames) == len(idmaps) == 4
    for arr in (*frames, *idmaps):
        assert arr.shape == (128, 128, 3) and arr.dtype == np.uint8
    # each dancer keeps its one colour across every frame of the clip
    for idmap in idmaps:
        present = decode(idmap, palette=pal, assignment=None)
        assert a[1] in present and a[2] in present


def test_occlusion_spec_window_contains_the_event(tmp_path):
    seq_dir = tmp_path / "dancetrack0002"
    _make_seq_with_gap(seq_dir, n=20)
    seq = load_sequence(seq_dir)
    specs = list(occlusion_specs(seq, length=10, min_gap=5, max_gap=40))
    assert specs, "expected at least one occlusion window for the mined gap"
    ev = specs[0].event
    assert ev is not None and ev.track_id == 1 and ev.gap == 6
    # the window straddles the gap: the last-before and first-after frames are inside it
    assert ev.before_frame in specs[0].frames and ev.after_frame in specs[0].frames


def test_iter_clips_yields_rendered_windows(tmp_path):
    root = tmp_path / "dancetrack" / "train1"
    _make_seq(root / "dancetrack0001", n=6)
    seqs = [load_sequence(root / "dancetrack0001")]
    clips = list(iter_clips(seqs, length=4, size=64, hop=4, include_occlusion=False))
    assert len(clips) == 1                      # one non-overlapping window fits in 6 frames
    frames, idmaps, spec = clips[0]
    assert spec.frames == [1, 2, 3, 4]
    assert len(frames) == len(idmaps) == 4 and frames[0].shape == (64, 64, 3)
