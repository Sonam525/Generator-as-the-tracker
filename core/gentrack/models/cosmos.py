"""CosmosNanoAdapter — a trained Cosmos3-Nano LoRA as a GeneratorAdapter for the
chained identity eval. Loads the base Nano pipeline, attaches the LoRA structure, and
loads the trained LoRA weights from a checkpoint (either a lora-only `lora.safetensors`
or a full `save_pretrained` dir — it extracts the `lora` keys either way).

`generate(conditioning_rgb) -> ID-map rgb` (uint8 HWC; output resized back to the input
size with NEAREST). Conditioning is i2v-spatial (validated 2026-06-14): the input frame is
packed as the clean condition latent (frame 0) and the ID-map is the denoised target, so
`num_frames` must match training (=5 → 2 latent frames) and the generated ID-map is the
LAST decoded frame (frame 0 is the RGB condition reconstruction). torch/diffusers/peft
imported lazily so the package stays importable without a GPU.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from gentrack.models.base import GeneratorAdapter

# Must match the P0/P1 training prompt (image_generators/cosmos/p0_finetune.py).
DEFAULT_PROMPT = (
    "segmentation ID map: paint each person a distinct flat colour on a black background"
)
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out", "gate_proj", "up_proj", "down_proj")


class CosmosNanoAdapter(GeneratorAdapter):
    def __init__(
        self,
        model_dir: str,
        lora_dir: str | None = None,
        *,
        prompt: str = DEFAULT_PROMPT,
        size: int = 1024,
        steps: int = 20,
        guidance: float = 4.0,
        seed: int = 0,
        r: int = 16,
        alpha: int = 16,
        num_frames: int = 5,
    ):
        self.model_dir = model_dir
        self.lora_dir = lora_dir
        self.prompt = prompt
        self.size = size
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.r = r
        self.alpha = alpha
        self.num_frames = num_frames
        self._pipe = None

    def _load_lora(self, tf) -> dict:
        import torch
        from peft import LoraConfig
        from safetensors.torch import load_file

        lin = sorted({n.split(".")[-1] for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)})
        targets = [n for n in LORA_TARGETS if n in lin]
        tf.add_adapter(LoraConfig(r=self.r, lora_alpha=self.alpha, target_modules=targets, lora_dropout=0.0))

        files = glob.glob(os.path.join(self.lora_dir, "*.safetensors"))
        sd = {}
        for shard in sorted(files):
            for k, v in load_file(shard).items():
                if "lora" in k:
                    sd[k] = v
        missing, unexpected = tf.load_state_dict(sd, strict=False)
        info = {"lora_tensors": len(sd), "unexpected": len(unexpected),
                "model_lora_keys": sum("lora" in n for n, _ in tf.named_parameters())}
        print(f"[CosmosNanoAdapter] {info}", flush=True)
        if sd and len(unexpected) == len(sd):
            raise RuntimeError("LoRA key names did not match the model — adapter NOT loaded")
        return info

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        pipe = Cosmos3OmniPipeline.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16, enable_safety_checker=False).to("cuda")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
        if self.lora_dir:
            self._load_lora(pipe.transformer)
        pipe.transformer.eval()
        self._pipe = pipe

    def generate(self, cond_rgb: np.ndarray) -> np.ndarray:
        self._ensure()
        import torch
        from PIL import Image

        h, w = cond_rgb.shape[:2]
        with torch.no_grad():
            o = self._pipe(
                prompt=self.prompt, image=Image.fromarray(np.asarray(cond_rgb, np.uint8)),
                num_frames=self.num_frames, height=self.size, width=self.size,
                num_inference_steps=self.steps, guidance_scale=self.guidance,
                generator=torch.Generator(device="cuda").manual_seed(self.seed))
        out = np.asarray(o.video[-1], dtype=np.uint8)  # frame 0 = RGB condition; ID-map is last
        if (out.shape[1], out.shape[0]) != (w, h):
            out = np.asarray(Image.fromarray(out).resize((w, h), Image.NEAREST), dtype=np.uint8)
        return out

    def frame_latent(self, rgb: np.ndarray) -> np.ndarray:
        """The frame's normalized VAE latent (z, size/16, size/16) for appearance pooling
        in the Option-3 re-id bank — encoded the same way as the i2v condition frame."""
        self._ensure()
        import torch
        from PIL import Image

        img = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).resize(
            (self.size, self.size), Image.BILINEAR))
        x = torch.from_numpy(img).permute(2, 0, 1)[None, :, None].float()
        x = (x / 127.5 - 1.0).to("cuda", torch.bfloat16)
        with torch.no_grad():
            lat = self._pipe._encode_video(x)            # (1, z, 1, h, w)
        return lat[0, :, 0].float().cpu().numpy()        # (z, h, w)

    def perceive(self, frame: np.ndarray) -> np.ndarray:
        return self.generate(frame)

    def edit(self, frame: np.ndarray, cond) -> np.ndarray:
        return self.generate(cond)
