"""Dataset ingestion against a synthetic MOTChallenge sequence fixture."""
import numpy as np
from PIL import Image

from gentrack.data.datasets import dancetrack_splits, load_sequence, load_split


def _make_seq(seq_dir, n_frames=3, w=160, h=80, with_info=True):
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True)
    for i in range(1, n_frames + 1):
        arr = np.full((h, w, 3), i * 10, dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{i:06d}.png")
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    lines = []
    for f in range(1, n_frames + 1):
        lines.append(f"{f},1,{10+f},20,30,40,1,1,1.0")
        lines.append(f"{f},2,{80+f},25,20,50,1,1,0.7")
    (gt_dir / "gt.txt").write_text("\n".join(lines) + "\n")
    if with_info:
        (seq_dir / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq_dir.name}\nimDir=img1\nframeRate=20\n"
            f"seqLength={n_frames}\nimWidth={w}\nimHeight={h}\nimExt=.png\n"
        )


def test_load_sequence(tmp_path):
    seq_dir = tmp_path / "dancetrack0001"
    _make_seq(seq_dir, n_frames=3)
    seq = load_sequence(seq_dir)
    assert seq.name == "dancetrack0001"
    assert seq.num_frames == 3
    assert seq.image_size == (160, 80)
    assert seq.track_ids == [1, 2]
    b = seq.boxes_at(2)
    assert len(b) == 2
    assert seq.frame_path(1).name == "000001.png"


def test_image_size_inferred_without_seqinfo(tmp_path):
    seq_dir = tmp_path / "seqNoInfo"
    _make_seq(seq_dir, n_frames=2, w=100, h=60, with_info=False)
    seq = load_sequence(seq_dir)
    assert seq.image_size == (100, 60)  # read from the PNG
    assert seq.info["name"] == "seqNoInfo"


def test_load_split(tmp_path):
    root = tmp_path / "dancetrack"
    for name in ("dancetrack0001", "dancetrack0002"):
        _make_seq(root / "train" / name, n_frames=2)
    seqs = load_split(root, "train")
    assert [s.name for s in seqs] == ["dancetrack0001", "dancetrack0002"]


def test_dancetrack_splits(tmp_path):
    root = tmp_path / "dancetrack"
    for d, names in [
        ("train1", ["dancetrack0001", "dancetrack0008"]),
        ("train2", ["dancetrack0052", "dancetrack0066"]),
        ("val", ["dancetrack0004"]),
    ]:
        for n in names:
            _make_seq(root / d / n, n_frames=2)
    train, val, test = dancetrack_splits(
        root, ["train1", "train2"], ["dancetrack0008", "dancetrack0066"], ["val"]
    )
    assert sorted(s.name for s in train) == ["dancetrack0001", "dancetrack0052"]
    assert sorted(s.name for s in val) == ["dancetrack0008", "dancetrack0066"]
    assert [s.name for s in test] == ["dancetrack0004"]


def test_load_split_multi_dir(tmp_path):
    """DanceTrack HF layout: train spread across train1/ + train2/."""
    root = tmp_path / "dancetrack"
    _make_seq(root / "train2" / "dancetrack0052", n_frames=2)
    _make_seq(root / "train1" / "dancetrack0001", n_frames=2)
    seqs = load_split(root, ["train1", "train2"])
    # globally sorted by sequence name regardless of which dir they came from
    assert [s.name for s in seqs] == ["dancetrack0001", "dancetrack0052"]
