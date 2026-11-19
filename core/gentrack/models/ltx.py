"""LTXVideoAdapter — Lightricks/LTX-2.3 (22B) as the video-mode (Option-C / P2-C)
generator: condition on an RGB clip, emit the aligned ID-map clip (v2v translation).

STATUS: SKELETON. LTX's native conditioning (`LTX2VideoCondition`) extends/interpolates
along time (condition frame i, generate frame j); our task is ALIGNED translation
(RGB frame t -> ID-map frame t). That spatially-aligned conditioning path is NOT a stock
pipeline (README §6.1 / §7.3). Which wiring to use —

  * channel_concat : concat the RGB latent with the noise latent per frame (ControlNet-
                     style structural control); frames stay 1:1 aligned, full window length;
                     needs a zero-init input-channel addition on the patch/proj_in.
  * temporal_pack  : pack [RGB clip | ID-map clip] as one sequence, condition the RGB half
                     via native LTX2VideoCondition(index=0), generate the ID-map half (the
                     direct video generalization of the validated Cosmos i2v trick).

— is DECIDED by video_generator/ablations/ltx_probe.py on the A100 via a self-IoU >> cross-IoU overfit (the
LTX analog of cond_sanity, which fixed the Cosmos i2v packing before it landed in
cosmos.py). Until the winner is ported, `generate_clip` raises.

torch/diffusers/peft are imported lazily so the package stays importable without a GPU.
The pure layout helpers below (frame budgeting, latent counts, temporal-pack split) are
CPU-testable and have no torch dependency.
"""
from __future__ import annotations

import numpy as np

from gentrack.models.base import GeneratorAdapter

# Must match the v2v training prompt (scripts/ltx_finetune.py, TBD) — same task statement
# as the image arm so a Stage-1 LoRA could plausibly transfer (README §6.1).
DEFAULT_PROMPT = (
    "segmentation ID map: paint each person a distinct flat colour on a black background"
)
# CONFIRMED by video_generator/ablations/ltx_probe.py: of the original guesses only QKV matched LTX-2.3's Linear
# leaves; attn-out is to_out.0 (leaf "0") and the FFN leaves are proj/linear/linear_1/linear_2 —
# ambiguous by leaf-name, so QKV-only until ltx_cond_sanity.py dumps full module paths to pin
# attn-out + FFN targets safely. QKV-only is 80M trainable and fit/trained fine in the probe.
LORA_TARGETS = ("to_q", "to_k", "to_v")

# FLAG UNVERIFIED until the probe confirms it on the real VAE (LTX-Video used 8x temporal /
# 32x spatial; LTX-2.3 unconfirmed). A wrong temporal ratio silently corrupts clip packing.
DEFAULT_TEMPORAL_RATIO = 8


# --- pure, CPU-testable layout helpers -------------------------------------------------

def valid_ltx_length(n_frames: int) -> bool:
    """LTX requires a pixel-frame count of the form 8k+1 (n % 8 == 1)."""
    return n_frames >= 1 and (n_frames - 1) % 8 == 0


def latent_frames(n_pixel: int, temporal_ratio: int = DEFAULT_TEMPORAL_RATIO) -> int:
    """Latent frame count for a causal video VAE: the first pixel frame maps to one latent
    frame, then every `temporal_ratio` further pixel frames add one (e.g. ratio 8:
    1->1, 9->2, 49->7, 121->16)."""
    return (n_pixel - 1) // temporal_ratio + 1


def temporal_pack_split(n_latent: int) -> tuple[list[int], list[int]]:
    """For temporal_pack mode: a length-`2*n_latent` sequence is laid out as
    [RGB-condition latent frames | ID-map-target latent frames]. Returns the
    (condition_indices, target_indices) — disjoint and contiguous."""
    return list(range(n_latent)), list(range(n_latent, 2 * n_latent))


# --- adapter ---------------------------------------------------------------------------

class LTXVideoAdapter(GeneratorAdapter):
    """Video-mode adapter. The eval harness only needs `temporal(window) -> [idmaps]`;
    per-frame `perceive`/`edit` are image-mode concepts and are not supported here.
    """

    def __init__(
        self,
        model_dir: str,
        lora_dir: str | None = None,
        *,
        prompt: str = DEFAULT_PROMPT,
        size: int = 512,
        frames: int = 49,
        steps: int = 20,
        guidance: float = 3.0,
        seed: int = 0,
        r: int = 16,
        alpha: int = 16,
        cond_mode: str = "channel_concat",
        temporal_ratio: int = DEFAULT_TEMPORAL_RATIO,
        disable_audio: bool = True,
    ):
        if not valid_ltx_length(frames):
            raise ValueError(f"frames={frames} invalid for LTX (need 8k+1, e.g. 49, 121)")
        if size % 32 != 0:
            raise ValueError(f"size={size} invalid for LTX (height/width must be % 32 == 0)")
        if cond_mode not in ("channel_concat", "temporal_pack"):
            raise ValueError(f"cond_mode={cond_mode!r} must be channel_concat | temporal_pack")
        self.model_dir = model_dir
        self.lora_dir = lora_dir
        self.prompt = prompt
        self.size = size
        self.frames = frames
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.r = r
        self.alpha = alpha
        self.cond_mode = cond_mode
        self.temporal_ratio = temporal_ratio
        self.disable_audio = disable_audio
        self._pipe = None

    def _load_lora(self, tf) -> dict:
        """Discover the transformer's Linear module names, intersect with LORA_TARGETS,
        attach the adapter, and load trained weights — same generic path as cosmos.py
        (the exact LTX-2.3 module names are confirmed at calibration, FLAG)."""
        import glob
        import os

        import torch
        from peft import LoraConfig
        from safetensors.torch import load_file

        lin = sorted({n.split(".")[-1] for n, m in tf.named_modules()
                      if isinstance(m, torch.nn.Linear)})
        targets = [n for n in LORA_TARGETS if n in lin]
        if not targets:
            raise RuntimeError(f"no LORA_TARGETS found among the transformer's Linear modules: {lin}")
        tf.add_adapter(LoraConfig(r=self.r, lora_alpha=self.alpha, target_modules=targets,
                                  lora_dropout=0.0))
        sd = {}
        for shard in sorted(glob.glob(os.path.join(self.lora_dir, "*.safetensors"))):
            for k, v in load_file(shard).items():
                if "lora" in k:
                    sd[k] = v
        _, unexpected = tf.load_state_dict(sd, strict=False)
        if sd and len(unexpected) == len(sd):
            raise RuntimeError("LoRA key names did not match the model — adapter NOT loaded")
        info = {"lora_tensors": len(sd), "targets": targets}
        print(f"[LTXVideoAdapter] {info}", flush=True)
        return info

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import LTX2ConditionPipeline

        pipe = LTX2ConditionPipeline.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        # FLAG: confirm the audio-disable API at calibration (drop the audio tower / pass a
        # flag) — we never decode audio, it is wasted VRAM at 22B.
        if self.lora_dir:
            self._load_lora(pipe.transformer)
        pipe.transformer.eval()
        self._pipe = pipe

    def generate_clip(self, rgb_clip: list[np.ndarray]) -> list[np.ndarray]:
        """RGB clip (list of uint8 HWC) -> aligned ID-map clip (same length, uint8 HWC).

        Conditioning is decided by video_generator/ablations/ltx_probe.py (channel_concat vs temporal_pack);
        the winning implementation is ported here. Until then this raises — writing an
        unvalidated conditioning path would repeat the §6.1 "stock pipeline" assumption the
        plan warns against.
        """
        raise NotImplementedError(
            f"cond_mode={self.cond_mode!r} not yet ported — run video_generator/ablations/ltx_probe.py first "
            "(self-IoU >> cross decides channel_concat vs temporal_pack)")

    def temporal(self, window: list[np.ndarray]) -> list[np.ndarray]:
        return self.generate_clip(window)

    def perceive(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LTXVideoAdapter is video-mode; use temporal()/generate_clip()")

    def edit(self, frame: np.ndarray, cond: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LTXVideoAdapter is video-mode; use temporal()/generate_clip()")
