"""Contact sheets for visual diagnosis: one PNG per chunk of frames, each
row a frame's (frame | GT map | generated map) so a single image reviews a whole
chain; report findings in text with these file paths.

Images are resized with NEAREST so flat ID-map colours stay crisp (and decodable
if re-read). Output is always lossless PNG.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _resize(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(img, dtype=np.uint8), "RGB").resize(size_wh, Image.NEAREST),
        dtype=np.uint8,
    )


def make_contact_sheet(
    rows: list[list[np.ndarray]],
    out_path,
    titles: tuple[str, ...] | None = ("frame", "GT", "generated"),
    tile_h: int = 140,
    pad: int = 3,
    bg: int = 50,
    label_h: int = 16,
) -> np.ndarray:
    """Tile `rows` (each a list of same-size RGB images) into one PNG grid.

    Returns the sheet array and writes it to `out_path` (must be .png).
    """
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        raise ValueError(f"contact sheet must be .png, got {out_path.suffix!r}")
    if not rows:
        raise ValueError("no rows to render")

    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    first = rows[0][0]
    tw = int(round(tile_h * first.shape[1] / first.shape[0]))
    title_h = label_h if titles else 0

    H = title_h + n_rows * (tile_h + pad) + pad
    W = n_cols * (tw + pad) + pad
    sheet = np.full((H, W, 3), bg, dtype=np.uint8)

    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            y = title_h + pad + r * (tile_h + pad)
            x = pad + c * (tw + pad)
            sheet[y : y + tile_h, x : x + tw] = _resize(img, (tw, tile_h))

    pil = Image.fromarray(sheet, "RGB")
    if titles:
        draw = ImageDraw.Draw(pil)
        for c, t in enumerate(titles[:n_cols]):
            draw.text((pad + c * (tw + pad) + 2, 2), t, fill=(230, 230, 230))
    pil.save(out_path, format="PNG")
    return np.asarray(pil, dtype=np.uint8)


def chunked_sheets(rows, out_dir, prefix="sheet", per_sheet: int = 10, **kw) -> list[Path]:
    """Split `rows` into chunks of `per_sheet` and write one PNG each."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(0, len(rows), per_sheet):
        p = out_dir / f"{prefix}_{i:04d}.png"
        make_contact_sheet(rows[i : i + per_sheet], p, **kw)
        paths.append(p)
    return paths
