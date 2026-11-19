"""MOTChallenge I/O: parsing, round-trip stability, grouping, seqinfo."""
from gentrack.data.mot_format import (
    Box,
    group_by_frame,
    group_by_track,
    read_gt,
    read_seqinfo,
    write_tracks,
)

GT_TEXT = """\
1,1,10.0,20.0,30.0,40.0,1,1,1.0
1,2,100.5,50.0,25.0,60.0,1,1,0.5
2,1,12.0,22.0,30.0,40.0,1,1,0.9
2,2,102.0,52.0,25.0,60.0,0,1,0.0
"""


def test_read_gt_full_columns(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text(GT_TEXT)
    boxes = read_gt(p)
    assert len(boxes) == 4
    b = boxes[1]
    assert (b.frame, b.track_id) == (1, 2)
    assert b.xywh == (100.5, 50.0, 25.0, 60.0)
    assert b.vis == 0.5
    assert boxes[3].conf == 0.0 and boxes[3].vis == 0.0


def test_read_gt_minimal_columns(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("5,7,1,2,3,4\n")  # only 6 columns
    (b,) = read_gt(p)
    assert (b.frame, b.track_id, b.conf, b.cls, b.vis) == (5, 7, 1.0, 1, 1.0)


def test_write_read_roundtrip(tmp_path):
    boxes = [
        Box(2, 5, 1, 2, 3, 4, conf=0.9),
        Box(1, 9, 5, 6, 7, 8, conf=0.5),
        Box(1, 2, 9, 9, 9, 9, conf=1.0),
    ]
    out = tmp_path / "res" / "seq.txt"
    write_tracks(boxes, out)
    text = out.read_text().strip().splitlines()
    # sorted by (frame, id)
    assert text[0].startswith("1,2,") and text[1].startswith("1,9,") and text[2].startswith("2,5,")
    reloaded = read_gt(out)
    assert {(b.frame, b.track_id) for b in reloaded} == {(2, 5), (1, 9), (1, 2)}
    assert reloaded[0].xywh == (9.0, 9.0, 9.0, 9.0)


def test_grouping(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text(GT_TEXT)
    boxes = read_gt(p)
    by_f = group_by_frame(boxes)
    assert set(by_f) == {1, 2} and len(by_f[1]) == 2
    by_t = group_by_track(boxes)
    assert set(by_t) == {1, 2}
    assert [b.frame for b in by_t[1]] == [1, 2]  # sorted by frame


def test_read_seqinfo(tmp_path):
    p = tmp_path / "seqinfo.ini"
    p.write_text(
        "[Sequence]\nname=demo\nimDir=img1\nframeRate=20\n"
        "seqLength=3\nimWidth=160\nimHeight=80\nimExt=.png\n"
    )
    info = read_seqinfo(p)
    assert info["name"] == "demo"
    assert info["imWidth"] == 160 and info["imHeight"] == 80
    assert info["seqLength"] == 3 and info["frameRate"] == 20
