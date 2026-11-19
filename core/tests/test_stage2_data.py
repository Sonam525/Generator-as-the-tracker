"""Stage-2 temporal pairs: shapes, stride, and sequence-stable colours."""
import numpy as np
from PIL import Image

from gentrack.color_codec import decode
from gentrack.data.build_idmaps import sequence_assignment
from gentrack.data.datasets import load_sequence
from gentrack.palette import build
from gentrack.train.stage2_data import iter_stage2, stage2_pairs_for_seq


def _make_seq(seq_dir, n=4, w=160, h=80):
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


def test_stage2_pairs_shapes_and_stride(tmp_path):
    seq_dir = tmp_path / "dancetrack0001"
    _make_seq(seq_dir, n=4)
    seq = load_sequence(seq_dir)
    pairs = list(stage2_pairs_for_seq(seq, build(32), size=128, stride=1))
    assert len(pairs) == 3  # n - stride
    ft, it, ftk, itk = pairs[0]
    for a in (ft, it, ftk, itk):
        assert a.shape == (128, 128, 3) and a.dtype == np.uint8
    assert len(list(stage2_pairs_for_seq(seq, build(32), size=128, stride=2))) == 2


def test_colours_are_sequence_stable(tmp_path):
    seq_dir = tmp_path / "dancetrack0001"
    _make_seq(seq_dir, n=4)
    seq = load_sequence(seq_dir)
    pal = build(32)
    a = sequence_assignment(seq.track_ids, 32)
    ft, it, ftk, itk = next(stage2_pairs_for_seq(seq, pal, size=128, stride=1))
    # the same dancer keeps the same palette colour across t and t+1
    free_t = decode(it, palette=pal, assignment=None)
    free_tk = decode(itk, palette=pal, assignment=None)
    assert a[1] in free_t and a[1] in free_tk   # id 1's colour present in both frames
    assert a[2] in free_t and a[2] in free_tk   # id 2's colour present in both frames


def test_iter_stage2_spans_strides(tmp_path):
    root = tmp_path / "dancetrack" / "train1"
    _make_seq(root / "dancetrack0001", n=4)
    seqs = [load_sequence(root / "dancetrack0001")]
    pairs = list(iter_stage2(seqs, size=64, strides=(1, 2)))
    assert len(pairs) == 3 + 2  # stride1: 3, stride2: 2
