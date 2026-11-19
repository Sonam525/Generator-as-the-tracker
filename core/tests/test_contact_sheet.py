"""Contact-sheet tiling."""
import numpy as np
import pytest

from gentrack.eval.contact_sheet import chunked_sheets, make_contact_sheet


def _img(c):
    return np.full((40, 60, 3), c, dtype=np.uint8)


def test_make_contact_sheet(tmp_path):
    rows = [[_img(50), _img(100), _img(150)] for _ in range(3)]
    p = tmp_path / "s.png"
    sheet = make_contact_sheet(rows, p)
    assert p.exists()
    assert sheet.ndim == 3 and sheet.shape[2] == 3


def test_refuses_non_png(tmp_path):
    with pytest.raises(ValueError):
        make_contact_sheet([[_img(0)]], tmp_path / "s.jpg")


def test_chunked(tmp_path):
    rows = [[_img(i % 256)] for i in range(25)]
    paths = chunked_sheets(rows, tmp_path, per_sheet=10)
    assert len(paths) == 3
    assert all(p.exists() for p in paths)
