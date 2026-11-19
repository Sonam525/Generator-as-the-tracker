"""MOT identity metrics (motmetrics wrapper) + ID-map → boxes export."""
import numpy as np

from gentrack.color_codec import encode
from gentrack.data.build_idmaps import OcclusionEvent
from gentrack.data.mot_format import Box
from gentrack.eval.mot_metrics import detection_recall, evaluate, reacquisition_check
from gentrack.palette import assign, build
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes


def _b(f, i, x, y, w=10, h=10):
    return Box(f, i, x, y, w, h)


def test_perfect_tracking():
    gt = [_b(1, 1, 0, 0), _b(1, 2, 50, 50), _b(2, 1, 1, 0), _b(2, 2, 51, 50)]
    pred = [_b(1, 1, 0, 0), _b(1, 2, 50, 50), _b(2, 1, 1, 0), _b(2, 2, 51, 50)]
    m = evaluate(gt, pred)
    assert m["idf1"] == 1.0
    assert m["mota"] == 1.0
    assert m["num_switches"] == 0


def test_id_swap_counts_switch():
    gt = [_b(1, 1, 0, 0), _b(1, 2, 50, 50), _b(2, 1, 1, 0), _b(2, 2, 51, 50)]
    pred = [_b(1, 1, 0, 0), _b(1, 2, 50, 50), _b(2, 2, 1, 0), _b(2, 1, 51, 50)]  # swap at f2
    m = evaluate(gt, pred)
    assert m["num_switches"] >= 1
    assert m["idf1"] < 1.0


def test_missed_detection_lowers_mota():
    gt = [_b(1, 1, 0, 0), _b(1, 2, 50, 50)]
    pred = [_b(1, 1, 0, 0)]  # id 2 missed
    m = evaluate(gt, pred)
    assert m["num_misses"] >= 1
    assert m["mota"] < 1.0


def test_reacquisition_check():
    # GT track 7 present f1 (before gap) and f10 (after gap); event gap covers 2..9
    gt = [_b(1, 7, 10, 10), _b(10, 7, 14, 12)]
    ev = OcclusionEvent(track_id=7, before_frame=1, after_frame=10, gap=8, kind="absence")

    # success: same pred id (99) overlaps GT 7 before and after the gap
    pred_ok = [_b(1, 99, 10, 10), _b(10, 99, 14, 12)]
    r = reacquisition_check(pred_ok, gt, [ev])
    assert r["n"] == 1 and r["reacq_rate"] == 1.0 and r["redetect_rate"] == 1.0
    assert r["events"][0]["reacquired"] is True

    # failure: different pred id after the gap (identity switch across occlusion)
    pred_switch = [_b(1, 99, 10, 10), _b(10, 42, 14, 12)]
    r2 = reacquisition_check(pred_switch, gt, [ev])
    assert r2["reacq_rate"] == 0.0 and r2["redetect_rate"] == 1.0  # re-detected but not re-acquired

    # not re-detected: no pred overlaps GT 7 at the reappearance frame
    pred_miss = [_b(1, 99, 10, 10), _b(10, 99, 500, 500)]
    r3 = reacquisition_check(pred_miss, gt, [ev])
    assert r3["redetect_rate"] == 0.0 and r3["reacq_rate"] == 0.0


def test_detection_recall():
    # 2 GT boxes in frame 1; a pred that exactly covers GT 1 only
    gt = [_b(1, 1, 0, 0, 10, 10), _b(1, 2, 50, 50, 10, 10)]
    pred = [_b(1, 99, 0, 0, 10, 10)]
    r = detection_recall(gt, pred, iou_thrs=(0.5, 0.1))
    assert r["n_gt"] == 2
    assert r["recall@0.5"] == 0.5 and r["recall@0.1"] == 0.5  # only GT 1 covered

    # add a pred partly overlapping GT 2 (IoU ~0.14): counts at 0.1 but not 0.5
    pred2 = pred + [_b(1, 88, 55, 55, 10, 10)]
    r2 = detection_recall(gt, pred2, iou_thrs=(0.5, 0.1))
    assert r2["recall@0.5"] == 0.5 and r2["recall@0.1"] == 1.0


def test_idmaps_to_boxes():
    pal, a = build(32), assign([5, 9], 32)  # canonical ids 5,9 -> palette idx 1,2
    m1 = np.zeros((40, 60), dtype=bool)
    m1[5:15, 5:20] = True
    m2 = np.zeros((40, 60), dtype=bool)
    m2[20:35, 30:50] = True
    idmap = encode([m1, m2], [5, 9], palette=pal, assignment=a)
    boxes = idmaps_to_boxes([idmap], palette=pal)
    assert len(boxes) == 2
    assert {b.track_id for b in boxes} == {1, 2}  # palette indices
    assert all(b.frame == 1 for b in boxes)
    by_id = {b.track_id: b for b in boxes}
    assert by_id[1].xywh == (5.0, 5.0, 15.0, 10.0)  # bbox of m1
