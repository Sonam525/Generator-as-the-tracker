"""Mine DanceTrack val gt.txt files for re-acquisition events (absence gaps).
Usage: python datasets/mine_occlusion_val.py <gt_dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from gentrack.data.build_idmaps import mine_occlusion  # noqa: E402
from gentrack.data.mot_format import read_gt  # noqa: E402

gt_dir = Path(sys.argv[1])
rows = []
for f in sorted(gt_dir.glob("*.txt")):
    boxes = read_gt(f)
    n_frames = max((b.frame for b in boxes), default=0)
    for e in mine_occlusion(boxes, strategy="absence", min_gap=5, max_gap=40):
        rows.append((f.stem, e.track_id, e.before_frame, e.after_frame, e.gap, n_frames))

# earliest reappearance first (shortest, cheapest run window)
rows.sort(key=lambda r: (r[3], r[4]))
print(f"total absence events (gap 5-40): {len(rows)}")
print(f"{'seq':16} {'track':5} {'before_f':8} {'after_f':7} {'gap':4} {'seq_len':7}")
for r in rows[:25]:
    print(f"{r[0]:16} {r[1]:<5} {r[2]:<8} {r[3]:<7} {r[4]:<4} {r[5]:<7}")
