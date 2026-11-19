"""Option-A memory canvas: best-crop capture, trails, grid canvas, conditioning."""
import numpy as np

from gentrack.palette import assign, build
from gentrack.pipeline.memory_canvas import MemoryCanvas

H, W = 80, 120


def _has_color(img, color):
    return bool(np.any(np.all(img == np.asarray(color), axis=-1)))


def _canvas():
    pal = build(32)
    assignment = assign([1, 2], 32)
    return MemoryCanvas(palette=pal, assignment=assignment), pal, assignment


def test_update_captures_best_crop_and_centroid():
    mc, pal, a = _canvas()
    frame = np.full((H, W, 3), 200, dtype=np.uint8)
    small = np.zeros((H, W), dtype=bool)
    small[10:20, 10:20] = True
    big = np.zeros((H, W), dtype=bool)
    big[10:40, 10:50] = True

    mc.update(frame, {1: small})
    assert mc.tracks[1].best_area == 100
    assert len(mc.tracks[1].centroids) == 1

    mc.update(frame, {1: big})  # larger -> replaces best crop
    assert mc.tracks[1].best_area == 30 * 40
    assert mc.tracks[1].best_crop.shape == (30, 40, 3)
    assert len(mc.tracks[1].centroids) == 2


def test_trail_capped():
    mc, _, _ = MemoryCanvas(max_trail=3), None, None
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    m = np.zeros((H, W), dtype=bool)
    m[5:8, 5:8] = True
    for _ in range(5):
        mc.update(frame, {7: m})
    assert len(mc.tracks[7].centroids) == 3  # capped


def test_build_canvas_shows_swatch_colors():
    mc, pal, a = _canvas()
    frame = np.full((H, W, 3), 150, dtype=np.uint8)
    m1 = np.zeros((H, W), dtype=bool)
    m1[5:25, 5:25] = True
    m2 = np.zeros((H, W), dtype=bool)
    m2[5:25, 60:80] = True
    mc.update(frame, {1: m1, 2: m2})

    canvas = mc.build_canvas()
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    # each id's palette swatch colour must appear somewhere in the canvas
    assert _has_color(canvas, pal[a[1]])
    assert _has_color(canvas, pal[a[2]])


def test_render_trails_paints_color():
    mc, pal, a = _canvas()
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    m = np.zeros((H, W), dtype=bool)
    m[40:50, 40:50] = True
    mc.update(frame, {1: m})
    trails = mc.render_trails((W, H))
    assert trails.shape == (H, W, 3)
    assert trails.any()  # something was painted at the centroid


def test_conditioning_reference_list():
    mc, pal, a = _canvas()
    frame = np.full((H, W, 3), 120, dtype=np.uint8)
    m = np.zeros((H, W), dtype=bool)
    m[5:25, 5:25] = True
    mc.update(frame, {1: m})
    prev_frame = np.full((H, W, 3), 80, dtype=np.uint8)
    prev_idmap = np.zeros((H, W, 3), dtype=np.uint8)
    refs = mc.conditioning(frame, prev_frame, prev_idmap, with_trails=True)
    assert len(refs) == 3  # composite + canvas + trails
    assert refs[0].shape == (H, W, 3)  # paired composite matches frame size
    assert all(r.ndim == 3 for r in refs)
    assert len(refs) <= 14
