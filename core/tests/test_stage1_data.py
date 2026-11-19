"""Stage-1 (frame -> ID-map) pair generation."""
import numpy as np
from PIL import Image

from gentrack.color_codec import decode
from gentrack.data.datasets import load_sequence
from gentrack.palette import assign, build
from gentrack.train.stage1_data import iter_stage1, stage1_pairs_for_seq


def _make_seq(seq_dir, n=3, w=160, h=80):
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        Image.fromarray(np.full((h, w, 3), 120, dtype=np.uint8)).save(img_dir / f"{i:06d}.png")
    gt = seq_dir / "gt"
    gt.mkdir()
    lines = []
    for f in range(1, n + 1):
        lines.append(f"{f},1,10,10,40,50,1,1,1")
        lines.append(f"{f},2,100,10,40,50,1,1,1")
    (gt / "gt.txt").write_text("\n".join(lines) + "\n")
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nseqLength={n}\n"
        f"imWidth={w}\nimHeight={h}\nimExt=.png\n"
    )


def test_stage1_pairs_shapes_and_decode(tmp_path):
    seq_dir = tmp_path / "dancetrack0001"
    _make_seq(seq_dir, n=3)
    seq = load_sequence(seq_dir)
    size = 128
    palette = build(32)
    pairs = list(stage1_pairs_for_seq(seq, palette, size=size))
    assert len(pairs) == 3
    frame, idmap = pairs[0]
    assert frame.shape == (size, size, 3) and frame.dtype == np.uint8
    assert idmap.shape == (size, size, 3)
    # ID-map decodes to the 2 tracks (boxes scaled into the square)
    rec = decode(idmap, palette=palette, assignment=assign([1, 2], 32))
    assert set(rec.keys()) == {1, 2}


def test_iter_stage1_spans_sequences(tmp_path):
    root = tmp_path / "dancetrack" / "train1"
    for name in ("dancetrack0001", "dancetrack0002"):
        _make_seq(root / name, n=2)
    seqs = [load_sequence(root / "dancetrack0001"), load_sequence(root / "dancetrack0002")]
    pairs = list(iter_stage1(seqs, size=64))
    assert len(pairs) == 4  # 2 seqs x 2 frames
    assert all(f.shape == (64, 64, 3) and m.shape == (64, 64, 3) for f, m in pairs)
