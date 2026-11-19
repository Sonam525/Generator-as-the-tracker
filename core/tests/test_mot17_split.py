"""MOT17 frame-half split: self-contained, re-indexed halves with preserved identities."""
from pathlib import Path

from gentrack.data.datasets import Sequence, frame_half
from gentrack.data.mot_format import Box


def _seq(n_frames=10, tracks=(1, 2)):
    frames = [Path(f"img1/{i:06d}.jpg") for i in range(1, n_frames + 1)]
    gt = [Box(f, t, 0, 0, 10, 10) for f in range(1, n_frames + 1) for t in tracks]
    return Sequence(name="MOT17-02-FRCNN", root=Path("/x"), frames=frames,
                    gt=gt, info={"imWidth": 100, "imHeight": 100})


def test_frame_half_splits_and_reindexes():
    th, vh = frame_half(_seq(10))
    assert th.num_frames == 5 and vh.num_frames == 5
    # halves re-index from 1 over the right original frames
    assert th.frame_path(1).name == "000001.jpg"
    assert vh.frame_path(1).name == "000006.jpg"
    assert vh.frame_path(5).name == "000010.jpg"


def test_frame_half_remaps_gt_and_keeps_identities():
    th, vh = frame_half(_seq(10, tracks=(7, 9)))
    # val-half frame 1 carries original frame-6 boxes, ids intact
    assert {b.track_id for b in vh.boxes_at(1)} == {7, 9}
    assert vh.boxes_at(6) == []                 # val-half only spans 5 frames
    assert set(th.track_ids) == {7, 9} and set(vh.track_ids) == {7, 9}


def test_frame_half_odd_length():
    th, vh = frame_half(_seq(11))
    assert th.num_frames == 5 and vh.num_frames == 6   # mid = 11//2 = 5
    assert vh.frame_path(1).name == "000006.jpg"
