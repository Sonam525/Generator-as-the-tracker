"""Pure helpers of data/pseudo_masks.py (no torch/SAM): match SAM masks to GT boxes,
paint the uint16 instance map, reconstruct masks. The Sam3Masker GPU path is smoke-
tested separately on the cluster."""
import numpy as np

from gentrack.data.mot_format import Box
from gentrack.data.pseudo_masks import (
    build_instance_map,
    idmap_from_instance_map,
    load_instance_map,
    mask_bbox,
    masks_from_instance_map,
    match_masks_to_boxes,
    resize_instance_map,
    save_instance_map,
)
from gentrack.palette import assign, build


def _mask(y0, y1, x0, x1, hw=(20, 20)):
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_bbox():
    assert mask_bbox(_mask(2, 7, 3, 9)) == (3.0, 2.0, 6.0, 5.0)
    assert mask_bbox(np.zeros((5, 5), dtype=bool)) is None


def test_match_masks_to_boxes_assigns_gt_identity():
    a = Box(1, 7, 0, 0, 10, 10)
    b = Box(1, 9, 12, 0, 8, 10)
    mask_a = _mask(0, 10, 0, 10)      # overlaps box 7
    mask_b = _mask(0, 10, 12, 20)     # overlaps box 9
    stray = _mask(18, 20, 18, 20)     # overlaps nothing
    out = match_masks_to_boxes([mask_b, stray, mask_a], [a, b], iou_thr=0.5)
    assert set(out) == {7, 9}
    assert np.array_equal(out[7], mask_a)
    assert np.array_equal(out[9], mask_b)


def test_match_unmatched_box_absent():
    a = Box(1, 7, 0, 0, 10, 10)
    far = _mask(15, 20, 15, 20)
    out = match_masks_to_boxes([far], [a], iou_thr=0.5)
    assert out == {}


def test_build_instance_map_fallback_and_paint_order():
    # box 7 (low vis) has a SAM mask; box 9 (high vis) overlaps and falls back to filled
    a = Box(1, 7, 0, 0, 10, 10, vis=0.1)
    b = Box(1, 9, 5, 0, 10, 10, vis=0.9)
    matched = {7: _mask(0, 10, 0, 10)}
    arr = build_instance_map(matched, [a, b], size=(20, 20), fallback_filled=True)
    assert arr[5, 0] == 7          # left strip stays box 7's mask
    assert arr[5, 7] == 9          # overlap goes to the more-visible (front) box 9
    assert arr[5, 12] == 9         # box 9 filled-box fallback region
    assert arr[15, 15] == 0        # background


def test_instance_map_roundtrip():
    arr = np.zeros((10, 10), dtype=np.uint16)
    arr[0:4, 0:4] = 7
    arr[5:9, 5:9] = 12
    masks, ids = masks_from_instance_map(arr)
    assert set(ids) == {7, 12}
    by = dict(zip(ids, masks))
    assert by[7].sum() == 16 and by[12].sum() == 16


def test_save_load_uint16_png(tmp_path):
    arr = np.zeros((8, 8), dtype=np.uint16)
    arr[1:5, 1:5] = 300  # > 255: verify true 16-bit round-trip, not 8-bit truncation
    p = str(tmp_path / "imap.png")
    save_instance_map(arr, p)
    back = load_instance_map(p)
    assert back.dtype == np.uint16 and np.array_equal(back, arr)


def test_idmap_from_instance_map_matches_codec():
    pal = build(48)
    a = assign([7, 12], 48)
    arr = np.zeros((20, 20), dtype=np.uint16)
    arr[2:8, 2:8] = 7
    arr[10:16, 10:16] = 12
    idmap = idmap_from_instance_map(arr, pal, a)
    # the two instances decode to their assigned palette colours
    assert tuple(idmap[4, 4]) == tuple(pal[a[7]])
    assert tuple(idmap[12, 12]) == tuple(pal[a[12]])
    assert tuple(idmap[0, 0]) == (0, 0, 0)  # background


def test_idmap_per_frame_assignment_when_none():
    pal = build(48)
    arr = np.zeros((20, 20), dtype=np.uint16)
    arr[2:8, 2:8] = 30
    arr[10:16, 10:16] = 31
    idmap = idmap_from_instance_map(arr, pal, assignment=None, K=48)
    cols = {tuple(idmap[4, 4]), tuple(idmap[12, 12])}
    assert len(cols) == 2 and (0, 0, 0) not in cols


def test_resize_instance_map_nearest_preserves_ids():
    arr = np.zeros((4, 4), dtype=np.uint16)
    arr[0:2, 0:2] = 5
    arr[2:4, 2:4] = 9
    out = resize_instance_map(arr, 8)
    assert out.shape == (8, 8)
    assert set(np.unique(out).tolist()).issubset({0, 5, 9})  # no interpolated ids
    assert (out == 5).sum() > 0 and (out == 9).sum() > 0


def test_stage1_sam_pseudo_branch(tmp_path):
    from PIL import Image

    from gentrack.data.datasets import Sequence
    from gentrack.train.stage1_data import stage1_pairs_for_seq

    img_dir = tmp_path / "img1"
    img_dir.mkdir()
    Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8)).save(img_dir / "000001.jpg")
    seq = Sequence(name="seqA", root=tmp_path, frames=[img_dir / "000001.jpg"],
                   gt=[], info={"imWidth": 40, "imHeight": 30})

    pm = tmp_path / "pm" / "seqA"
    pm.mkdir(parents=True)
    arr = np.zeros((30, 40), dtype=np.uint16)
    arr[5:20, 5:20] = 7
    save_instance_map(arr, str(pm / "00000001.png"))

    pairs = list(stage1_pairs_for_seq(seq, build(48), size=32, mask_source="sam_pseudo",
                                      pseudo_root=str(tmp_path / "pm")))
    assert len(pairs) == 1
    frame, idmap = pairs[0]
    assert frame.shape == (32, 32, 3) and idmap.shape == (32, 32, 3)
    assert (idmap.reshape(-1, 3) != 0).any()  # silhouette painted
