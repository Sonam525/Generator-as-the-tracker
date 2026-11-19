"""SAM 3.1 (facebook/sam3) pseudo-masks — Mode 1: per-frame box-prompted image
segmentation with identity taken from GT, never from SAM (§8.2).

For each frame we prompt SAM with the frame's GT boxes, then match each returned
instance mask to the GT box it best overlaps and label it with that box's track id.
Identity is therefore provably correct even on adversarial near-identical content —
SAM only does what it is unbeatable at (box -> tight silhouette); GT supplies identity.
Unmatched GT boxes fall back to the filled box so no instance is ever dropped.

Output per frame: a uint16 instance map (pixel = track id, 0 = background), lossless
PNG, one file per frame. `masks_from_instance_map` reconstructs per-instance masks for
the codec at train time. Generated once over TRAIN splits only (val hygiene, rule 5).

torch/transformers imported lazily so the module stays importable without a GPU; only
`Sam3Masker` needs the cluster. Everything else is pure and unit-tested.
"""
from __future__ import annotations

import os

import numpy as np

from gentrack.data.mot_format import Box

SAM3_MODEL_ID = "facebook/sam3"


def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mask_bbox(mask: np.ndarray):
    """(x, y, w, h) of a boolean mask's tight bounding box, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))


def match_masks_to_boxes(masks, boxes, iou_thr: float = 0.5) -> dict[int, np.ndarray]:
    """Greedy match SAM masks -> GT boxes by bbox-IoU (each used once).

    Returns {track_id: mask} for matched boxes only; unmatched track ids are absent
    (caller decides the fallback). Robust to SAM returning more/fewer/extra masks than
    boxes and to whether box prompts behave geometrically or as exemplars — the GT box
    a mask overlaps decides its identity.
    """
    cand = [(m, mask_bbox(m)) for m in masks]
    cand = [(m, bb) for m, bb in cand if bb is not None]
    triples = []
    for bi, b in enumerate(boxes):
        for mi, (_m, bb) in enumerate(cand):
            iou = _iou_xywh(b.xywh, bb)
            if iou >= iou_thr:
                triples.append((iou, bi, mi))
    triples.sort(key=lambda t: t[0], reverse=True)
    used_b: set[int] = set()
    used_m: set[int] = set()
    out: dict[int, np.ndarray] = {}
    for _iou, bi, mi in triples:
        if bi in used_b or mi in used_m:
            continue
        used_b.add(bi)
        used_m.add(mi)
        out[boxes[bi].track_id] = cand[mi][0]
    return out


def build_instance_map(matched, boxes, size, fallback_filled: bool = True) -> np.ndarray:
    """Paint a uint16 instance map (pixel = track id, 0 = bg) from matched SAM masks.

    `size` = (width, height). Painted low-visibility first so the most-visible (front)
    instance ends up on top — same depth proxy as filled_box_masks. Boxes with no SAM
    mask are painted as filled boxes when `fallback_filled` (else skipped).
    """
    w, h = size
    arr = np.zeros((h, w), dtype=np.uint16)
    for b in sorted(boxes, key=lambda b: (b.vis, b.track_id)):
        m = matched.get(b.track_id)
        if m is not None:
            arr[m.astype(bool)] = b.track_id
        elif fallback_filled:
            x0 = max(0, int(round(b.x)))
            y0 = max(0, int(round(b.y)))
            x1 = min(w, int(round(b.x + b.w)))
            y1 = min(h, int(round(b.y + b.h)))
            if x1 > x0 and y1 > y0:
                arr[y0:y1, x0:x1] = b.track_id
    return arr


def masks_from_instance_map(arr: np.ndarray):
    """uint16 instance map -> (masks, ids) for the codec. 0 is background."""
    masks, ids = [], []
    for v in np.unique(arr):
        if v == 0:
            continue
        masks.append(arr == v)
        ids.append(int(v))
    return masks, ids


def save_instance_map(arr: np.ndarray, path: str) -> None:
    """Write a uint16 instance map as a lossless single-channel PNG (mode I;16)."""
    from PIL import Image

    Image.fromarray(arr.astype(np.uint16)).save(path)  # uint16 2D -> mode I;16


def load_instance_map(path: str) -> np.ndarray:
    """Read a uint16 instance map PNG back to an array."""
    from PIL import Image

    return np.asarray(Image.open(path), dtype=np.uint16)


def resize_instance_map(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a uint16 instance map to (size, size) with NEAREST — never interpolate
    (interpolation invents track ids between silhouettes). Matches the square frame
    resize used in training so frame and ID-map live in the same distorted space."""
    from PIL import Image

    im = Image.fromarray(arr.astype(np.uint16)).resize((size, size), Image.NEAREST)
    return np.asarray(im, dtype=np.uint16)


def idmap_from_instance_map(arr: np.ndarray, palette, assignment=None,
                            K: int | None = None) -> np.ndarray:
    """uint16 instance map -> RGB ID-map via the codec. `assignment` is an explicit
    id->colour map (sequence-stable colours, Stage-2); if None, colours are assigned
    per-frame from the ids present (Stage-1), mirroring render_idmap(assignment=None)."""
    from gentrack.color_codec import encode
    from gentrack.palette import DEFAULT_K, assign

    masks, ids = masks_from_instance_map(arr)
    if not masks:
        h, w = arr.shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    if assignment is None:
        assignment = assign(ids, K or DEFAULT_K)
    return encode(masks, ids, palette=palette, assignment=assignment)


def generate_for_sequence(seq, masker: "Sam3Masker", out_dir: str, *,
                          iou_thr: float = 0.5, resume: bool = True) -> dict:
    """Write one uint16 instance-map PNG per frame for a sequence. Resumable (skips
    frames already on disk). Returns coverage stats (frames, boxes, matched)."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    w, h = seq.image_size
    stats = {"frames": 0, "boxes": 0, "matched": 0, "skipped": 0}
    for f in range(1, seq.num_frames + 1):
        path = os.path.join(out_dir, f"{f:08d}.png")
        if resume and os.path.exists(path):
            stats["skipped"] += 1
            continue
        boxes = seq.boxes_at(f)
        img = np.asarray(Image.open(seq.frame_path(f)).convert("RGB"))
        masks = masker.masks_for(img, boxes)
        matched = match_masks_to_boxes(masks, boxes, iou_thr=iou_thr)
        save_instance_map(build_instance_map(matched, boxes, (w, h)), path)
        stats["frames"] += 1
        stats["boxes"] += len(boxes)
        stats["matched"] += len(matched)
    return stats


def _box_xyxy(b: Box):
    return [float(b.x), float(b.y), float(b.x + b.w), float(b.y + b.h)]


class Sam3Masker:
    """Lazy wrapper over facebook/sam3 (transformers Sam3Model/Sam3Processor) for
    box-prompted instance masks. `masks_for(image_rgb, boxes)` returns a list of
    boolean masks at the image's resolution — identity is NOT assigned here."""

    def __init__(
        self,
        model_id: str = SAM3_MODEL_ID,
        *,
        device: str = "cuda",
        mask_threshold: float = 0.5,
        score_threshold: float = 0.4,
        hf_token: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.mask_threshold = mask_threshold
        self.score_threshold = score_threshold
        self.hf_token = hf_token
        self._model = None
        self._proc = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        # float32: SAM3 is small (~3.45 GB), and bf16 cannot represent pixel-coordinate
        # box prompts above 256 exactly — casting boxes to bf16 corrupts the prompt
        # (the processor emits fp32 boxes; the bf16 weights then dtype-mismatch anyway).
        self._proc = Sam3Processor.from_pretrained(self.model_id, token=self.hf_token)
        self._model = Sam3Model.from_pretrained(
            self.model_id, torch_dtype=torch.float32, token=self.hf_token).to(self.device).eval()

    def masks_for(self, image_rgb: np.ndarray, boxes) -> list[np.ndarray]:
        """Box-prompt SAM with every GT box; return its instance masks (bool HxW).

        One processor call per frame (all boxes batched). `boxes` are mot_format.Box;
        only their geometry is used. Masks come back at the original image size.
        """
        self._ensure()
        import torch
        from PIL import Image

        if not boxes:
            return []
        img = Image.fromarray(np.asarray(image_rgb, np.uint8))
        xyxy = [_box_xyxy(b) for b in boxes]
        inputs = self._proc(
            images=img,
            input_boxes=[xyxy],
            input_boxes_labels=[[1] * len(xyxy)],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._proc.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        return _extract_masks(results[0])


def _extract_masks(result) -> list[np.ndarray]:
    """Pull boolean instance masks out of a post_process_instance_segmentation result,
    tolerant to the two common shapes: a {'masks': [N,H,W]} dict or a panoptic-style
    {'segmentation': HxW, 'segments_info': [...]} map."""
    masks: list[np.ndarray] = []
    if isinstance(result, dict) and result.get("masks") is not None:
        arr = result["masks"]
        arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)
        for m in arr:
            masks.append(np.asarray(m).astype(bool))
        return masks
    if isinstance(result, dict) and result.get("segmentation") is not None:
        seg = result["segmentation"]
        seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
        for v in np.unique(seg):
            if v == 0:
                continue
            masks.append(seg == v)
        return masks
    return masks
