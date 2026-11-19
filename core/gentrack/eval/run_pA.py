"""P-A — Option-A memory-canvas baseline runner (zero training; §13 GATE A).

Three measurements over DanceTrack val clips, all decode-based:
  (a) chain-length  : persistence half-life over a long consecutive chain
                       (with vs without the decode->correct->re-encode loop)
  (b) frame-gap     : persistence vs stride 1/2/4/8 (keyframe-scheduler viability)
  (c) occlusion     : re-acquisition rate after mined gaps (5/10/20)

GATE A output: the baseline curve every trained mechanism must beat (Gate 2C).
Cost is dollars, not GPUs — every generated frame is one paid NB2 image. The
spend gate (`confirm_spend`) refuses to run a batch over ~$10 without explicit
human approval (`assume_yes`), per rule 6 of the working agreement (README.md).

The data-driven `main` needs DanceTrack on the volume; the core functions are
unit-tested with a mock adapter (no network, no spend).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gentrack.color_codec import encode
from gentrack.data.build_idmaps import filled_box_masks, mine_occlusion, sequence_assignment
from gentrack.eval.contact_sheet import chunked_sheets
from gentrack.eval.persistence import half_life, reacquisition_rate, survival_curve
from gentrack.models.nanobanana_api import estimate_cost
from gentrack.palette import build
from gentrack.pipeline.memory_canvas import MemoryCanvas
from gentrack.pipeline.track import reencode, run_chain


class SpendNotConfirmed(RuntimeError):
    """Raised when an estimated batch cost exceeds the gate without approval."""


def confirm_spend(n_images: int, confirm_above: float = 10.0, assume_yes: bool = False) -> float:
    """Estimate batch cost; refuse (raise) above the gate unless approved."""
    cost = estimate_cost(n_images)
    if cost > confirm_above and not assume_yes:
        raise SpendNotConfirmed(
            f"estimated ${cost:.2f} for {n_images} images exceeds ${confirm_above:.0f}; "
            "get human approval, then pass assume_yes=True"
        )
    return cost


def sequence_gt_masks(seq) -> list[dict[int, np.ndarray]]:
    """Per-frame {id: filled-box mask} from a Sequence's GT (1-indexed frames)."""
    size = seq.image_size
    out = []
    for f in range(1, seq.num_frames + 1):
        masks, ids = filled_box_masks(seq.boxes_at(f), size)
        out.append({i: m for m, i in zip(masks, ids)})
    return out


def load_frames(seq, max_frames: int | None = None) -> list[np.ndarray]:
    n = seq.num_frames if max_frames is None else min(seq.num_frames, max_frames)
    frames = []
    for f in range(1, n + 1):
        with Image.open(seq.frame_path(f)) as im:
            frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return frames


def track_sequence(
    adapter,
    frames,
    gt_per_frame,
    *,
    palette,
    assignment,
    min_area: int = 16,
    use_canvas: bool = True,
    clean_loop: bool = True,
    iou_threshold: float = 0.5,
    contact_dir=None,
    contact_every: int = 10,
) -> dict:
    """Chained Option-A tracking + persistence metrics over one clip."""
    state = MemoryCanvas(palette=palette, assignment=assignment) if use_canvas else None
    clean = run_chain(
        adapter, frames, palette=palette, assignment=assignment,
        min_area=min_area, state=state, clean_loop=clean_loop,
    )
    curve = survival_curve(clean, gt_per_frame, palette, assignment, iou_threshold, min_area)
    result = {
        "survival_curve": curve,
        "half_life": half_life(curve),
        "cost_usd": getattr(adapter, "cost_usd", 0.0),
        "clean_maps": clean,
    }
    if contact_dir is not None:
        rows = []
        for frame, gt, cm in zip(frames, gt_per_frame, clean):
            gt_map = (
                encode(list(gt.values()), list(gt.keys()), palette=palette, assignment=assignment)
                if gt
                else np.zeros_like(cm)
            )
            rows.append([frame, gt_map, cm])
        result["contact_sheets"] = [
            str(p) for p in chunked_sheets(rows, contact_dir, per_sheet=contact_every)
        ]
    return result


def run_pa_clip(
    adapter,
    frames,
    gt_per_frame,
    *,
    palette,
    K: int = 32,
    min_area: int = 15000,
    use_canvas: bool = True,
    with_trails: bool = False,
    contact_dir=None,
    contact_every: int = 10,
) -> dict:
    """Real-data Option-A chain (validated config; finding gentrack-nb2-findings).

    Loop: perceive frame 0; for each next frame condition NB2 on CRISP references
    [prev frame, prev canonical ID-map, memory canvas (+trails)] → decode-free →
    CanonicalTracker matches blobs to stable canonical colours by IoU+colour →
    re-encode. `min_area` filters sub-person speckle (~0.7% of a 1080p frame at
    15000; scale with resolution).

    Produces canonical clean maps + contact sheets (frame | GT-box map | generated).
    Track ids are tracker-assigned (not GT ids) → GT-aligned IDF1/HOTA via
    TrackEval comes next; contact sheets + track-count stability are today's check.
    """
    from gentrack.palette import assign as _assign
    from gentrack.pipeline.canonicalize import CanonicalTracker

    tracker = CanonicalTracker(palette, K=K, min_area=min_area)
    canvas = MemoryCanvas(palette=palette) if use_canvas else None

    clean_maps: list = []
    prev_clean = prev_frame = None
    for frame in frames:
        if prev_clean is None:
            raw = adapter.perceive(frame)
        else:
            refs = [prev_frame, prev_clean]
            if canvas is not None:
                refs.append(canvas.build_canvas())
                if with_trails:
                    refs.append(canvas.render_trails((frame.shape[1], frame.shape[0])))
            raw = adapter.edit(frame, refs)

        inst = tracker.update(raw, frame)
        assignment = tracker.assignment()
        clean = reencode(inst, frame.shape[:2], palette, assignment)
        clean_maps.append(clean)
        if canvas is not None:
            canvas.assignment = assignment  # swatches use canonical colours
            canvas.update(frame, inst)
        prev_clean, prev_frame = clean, frame

    result = {
        "clean_maps": clean_maps,
        "n_tracks": len(tracker.tracks),
        "cost_usd": getattr(adapter, "cost_usd", 0.0),
    }
    if contact_dir is not None:
        gt_ids = sorted({i for g in gt_per_frame for i in g})
        ga = _assign(gt_ids, K) if gt_ids else {}
        rows = []
        for frame, g, cm in zip(frames, gt_per_frame, clean_maps):
            gt_map = (
                encode(list(g.values()), list(g.keys()), palette=palette, assignment=ga)
                if g
                else np.zeros_like(cm)
            )
            rows.append([frame, gt_map, cm])
        result["contact_sheets"] = [
            str(p) for p in chunked_sheets(rows, contact_dir, per_sheet=contact_every)
        ]
    return result


def evaluate_clip(clean_maps, seq, palette, iou_threshold: float = 0.5):
    """Box-IoU MOT metrics for a clip: decode canonical maps → pred boxes, compare
    to GT boxes for frames 1..len(clean_maps). Returns (metrics, pred_boxes, gt_boxes)."""
    from gentrack.eval.mot_metrics import evaluate
    from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes

    pred = idmaps_to_boxes(clean_maps, palette=palette)
    gt = [b for f in range(1, len(clean_maps) + 1) for b in seq.boxes_at(f)]
    return evaluate(gt, pred, iou_threshold), pred, gt


def gap_sweep(adapter, frames, gt_per_frame, *, strides, **kw) -> dict:
    """Persistence vs stride (keyframe scheduling). Skips contact sheets per stride."""
    kw.pop("contact_dir", None)
    out = {}
    for s in strides:
        r = track_sequence(adapter, frames[::s], gt_per_frame[::s], **kw)
        out[s] = {"half_life": r["half_life"], "survival_curve": r["survival_curve"]}
    return out


def main() -> int:
    import argparse

    import yaml

    from gentrack.data.datasets import load_sequence

    ap = argparse.ArgumentParser(description="P-A memory-canvas baseline (paid NB2).")
    ap.add_argument("--seq", required=True, help="path to a DanceTrack sequence dir")
    ap.add_argument("--out", required=True, help="output dir (on the volume)")
    ap.add_argument("--data-config", default="datasets/data.yaml")
    ap.add_argument("--eval-config", default="image_generators/nano_banana/eval.yaml")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--no-canvas", action="store_true")
    ap.add_argument("--no-clean-loop", action="store_true")
    ap.add_argument("--yes", action="store_true", help="approve estimated spend")
    args = ap.parse_args()

    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())
    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    min_area = eval_cfg["decode"]["min_area"]
    iou_thr = eval_cfg["persistence"]["iou_threshold"]
    strides = eval_cfg["pA"]["strides"]
    strategy = data_cfg["datasets"]["dancetrack"]["occlusion_strategy"]
    gmin, gmax = (
        data_cfg["occlusion_mining"]["min_gap"],
        data_cfg["occlusion_mining"]["max_gap"],
    )

    seq = load_sequence(args.seq)
    frames = load_frames(seq, args.max_frames)
    gt = sequence_gt_masks(seq)[: len(frames)]
    palette = build(data_cfg["palette"]["K"])
    assignment = sequence_assignment(seq.track_ids, data_cfg["palette"]["K"])

    # cost: one chain + the stride>1 sweeps
    n_images = len(frames) + sum(len(frames[::s]) for s in strides if s > 1)
    cost = confirm_spend(n_images, eval_cfg.get("confirm_above_usd", 10.0), args.yes)
    print(f"estimated spend: ${cost:.2f} for ~{n_images} images")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chain = track_sequence(
        adapter=_build_adapter(),
        frames=frames,
        gt_per_frame=gt,
        palette=palette,
        assignment=assignment,
        min_area=min_area,
        use_canvas=not args.no_canvas,
        clean_loop=not args.no_clean_loop,
        iou_threshold=iou_thr,
        contact_dir=out_dir / "contact",
    )
    events = mine_occlusion(seq.gt, strategy=strategy, min_gap=gmin, max_gap=gmax)
    occ = reacquisition_rate(
        chain["clean_maps"], gt, events, palette, assignment, iou_thr, min_area
    )
    sweep = gap_sweep(
        _build_adapter(), frames, gt, strides=strides, palette=palette,
        assignment=assignment, min_area=min_area, iou_threshold=iou_thr,
        use_canvas=not args.no_canvas, clean_loop=not args.no_clean_loop,
    )

    results = {
        "sequence": seq.name,
        "n_frames": len(frames),
        "chain_half_life": chain["half_life"],
        "chain_survival_curve": chain["survival_curve"],
        "gap_sweep": sweep,
        "occlusion": occ,
        "contact_sheets": chain.get("contact_sheets", []),
        "cost_usd": chain["cost_usd"],
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "chain_survival_curve"}, indent=2))
    print(f"\nresults + contact sheets in {out_dir}")
    return 0


def _build_adapter():
    from gentrack.models.nanobanana_api import NanoBananaAdapter

    return NanoBananaAdapter()


if __name__ == "__main__":
    raise SystemExit(main())
