# Databricks notebook source
# LTX-2.3 conditioning sanity — DECIDES cond_mode (channel_concat vs temporal_pack) for the
# aligned v2v task (RGB clip t -> ID-map clip t), the LTX analog of cond_sanity.py for Cosmos.
#
# Logic (same as Cosmos): overfit TWO 49-frame clips that share the IDENTICAL prompt; the only
# thing that can tell them apart is the RGB condition. Then reconstruct each from the SAME noise
# conditioned on its own RGB. Conditioning works iff self-IoU >> cross-IoU. We run this for both
# packings and the winner (higher self, bigger self-cross margin) is ported into
# models/ltx.py:generate_clip.
#
#   channel_concat : hidden_states = [rgb_latent | noisy_target_latent] on the CHANNEL dim
#                    (128->256); proj_in expanded (+128 zero-init) and made TRAINABLE so the
#                    RGB channels can be learned (ControlNet-style). 1:1 aligned, 7 latent frames.
#   temporal_pack  : hidden_states = [rgb_latent(clean) ; noisy_target_latent] on the TIME/token
#                    dim (1792->3584); cond tokens get timestep 0, target tokens timestep t; loss
#                    on the target half only. proj_in unchanged (QKV LoRA carries it via attention)
#                    — the direct generalization of the validated Cosmos i2v trick.
#
# STAGED + incremental saves (LTX's interface is richer than Cosmos': per-TOKEN timesteps, a
# parallel audio branch, RoPE video_coords, per-channel latent normalization). Each stage saves
# to results/ltx_cond_sanity/sanity.json so a first run still yields the VAE/normalization/
# scaffold facts even if the overfit needs a fix:
#   A. load (offload) + capture base transformer scaffolds at 7- and 14-latent-frame lengths;
#      dump FULL Linear module paths (video vs audio) to pin attn-out + FFN LoRA targets later.
#   B. latent normalization (vae.config mean/std/scaling) + VAE IDENTITY round-trip
#      (idmap -> vae.enc -> vae.dec -> codec decode -> IoU vs GT): does the VAE preserve flat
#      colours? A prerequisite for the whole v2v arm — if it smears them, that's a finding.
#   C. per-mode overfit (QKV LoRA, ~200 steps) + single-seed few-step reconstruction -> self/cross.

import json
import math
import os
import subprocess
import sys
import time
import traceback


def pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    HF_TOKEN = dbutils.secrets.get("gentrack", "HF_TOKEN")  # noqa: F821
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

pip("git+https://github.com/huggingface/diffusers.git", "accelerate", "transformers",
    "peft", "safetensors", "sentencepiece", "imageio", "scipy", "scikit-image")

sys.path.insert(0, os.environ["GENTRACK_WS"] + "/core")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from gentrack.data.build_clips import render_clip  # noqa: E402
from gentrack.data.datasets import load_sequence  # noqa: E402
from gentrack.palette import build  # noqa: E402
from gentrack.pipeline.decode_to_tracks import idmaps_to_boxes  # noqa: E402

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
MODEL = f"{VOL}/models/ltx2_3"
PSEUDO = f"{VOL}/pseudo_masks/dancetrack"
RES = f"{VOL}/results/ltx_cond_sanity"
os.makedirs(RES, exist_ok=True)

SIZE = 512
WIN = 49                 # pixel frames per clip -> 7 latent frames (ratio 8)
K = 48                   # palette (matches the project's K=48 CIELAB set)
STEPS = 300              # overfit steps per mode
LR = 2e-4
PROMPT = "segmentation ID map: paint each person a distinct flat colour on a black background"
# Two visually different windows of one sequence; identical prompt -> only RGB distinguishes them.
SEQ = "dancetrack0008"
STARTS = [1, 400]
palette = build(K)

res = {"size": SIZE, "win": WIN, "K": K, "seq": SEQ, "starts": STARTS, "steps": STEPS}


def save():
    with open(f"{RES}/sanity.json", "w") as f:
        json.dump(res, f, indent=2, default=str)


def vram_peak():
    return [round(torch.cuda.max_memory_allocated(i) / 1e9, 1)
            for i in range(torch.cuda.device_count())]


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mean_iou(pred_boxes, gt_boxes):
    if not gt_boxes:
        return None
    if not pred_boxes:
        return 0.0
    C = np.array([[box_iou(g.xywh, p.xywh) for p in pred_boxes] for g in gt_boxes])
    r, c = linear_sum_assignment(-C)
    return float(np.mean([C[i, j] for i, j in zip(r, c)]))


# ---- data: two (RGB clip, ID-map clip) windows, sequence-stable colours -----------------
seq = load_sequence(f"{VOL}/dancetrack/train1/{SEQ}")
clips = []  # each: {"rgb": [uint8 HWC]*WIN, "idmap": [uint8 HWC]*WIN, "gt_boxes": [...]}
for st in STARTS:
    frames = list(range(st, st + WIN))
    rgb, idmap = render_clip(seq, frames, palette, SIZE, K=K,
                             mask_source="sam_pseudo", pseudo_root=PSEUDO)
    mid = WIN // 2
    gt = idmaps_to_boxes([idmap[mid]], palette=palette, min_area=200)  # GT boxes at the mid frame
    clips.append({"rgb": rgb, "idmap": idmap, "gt": gt, "mid": mid})
res["gt_tracks"] = [len({b.track_id for b in c["gt"]}) for c in clips]
log(f"clips ready | gt tracks {res['gt_tracks']}")
save()

# ---- load + offload ----
from diffusers import LTX2ConditionPipeline  # noqa: E402

pipe = LTX2ConditionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
tf = pipe.transformer
vae = pipe.vae
log("pipeline loaded (cpu_offload)")

# ---- A. capture base scaffolds (hook-and-stop) at 7- and 14-latent-frame lengths ----
captured = {}


class _Stop(Exception):
    pass


def make_hook(key):
    def hook(module, args, kwargs):
        if key not in captured:
            captured[key] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            raise _Stop()
    return hook


def capture_scaffold(key, n_pixel):
    """One transformer forward of an unconditioned t2v gen at n_pixel frames -> kwargs scaffold."""
    h = tf.register_forward_pre_hook(make_hook(key), with_kwargs=True)
    try:
        with torch.no_grad():
            pipe(prompt=PROMPT, num_frames=n_pixel, height=SIZE, width=SIZE,
                 num_inference_steps=8, guidance_scale=1.0, output_type="latent",
                 return_dict=False, generator=torch.Generator(device="cuda").manual_seed(0))
    except _Stop:
        pass
    finally:
        h.remove()


def slice_b1(d):
    """Scaffold may come batched (CFG). Keep batch 1 (last row = cond) for tensors; pass scalars."""
    out = {}
    for k, v in d.items():
        out[k] = v[-1:].clone() if torch.is_tensor(v) and v.dim() >= 1 else v
    return out


try:
    capture_scaffold("f7", WIN)            # 49 px -> 7 latent (channel_concat)
    capture_scaffold("f14", 8 * 13 + 1)    # 105 px -> 14 latent (temporal_pack)
    sc7, sc14 = slice_b1(captured["f7"]), slice_b1(captured["f14"])
    res["scaffold_f7_keys"] = {k: (list(v.shape) if torch.is_tensor(v) else v) for k, v in sc7.items()}
    res["scaffold_f14_hidden"] = list(sc14["hidden_states"].shape)
    # derive the timestep<->sigma scaling from the captured step-0 values (timestep is per-token)
    t0 = float(sc7["timestep"].flatten()[0]); s0 = float(sc7["sigma"].flatten()[0])
    res["t_over_sigma"] = round(t0 / s0, 2) if s0 else None
    res["scaffold_t0"], res["scaffold_s0"] = t0, s0
    log(f"scaffolds captured | f7 hidden {res['scaffold_f7_keys'].get('hidden_states')} "
        f"f14 hidden {res['scaffold_f14_hidden']} | t/sigma {res['t_over_sigma']}")
    # FULL Linear module paths (video vs audio) to pin attn-out + FFN LoRA targets next
    lin = [n for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)]
    res["linear_paths_sample"] = [n for n in lin if ".0." in n or "blocks.0." in n or "layers.0." in n][:40]
    res["linear_paths_audio"] = [n for n in lin if "audio" in n.lower()][:10]
    save()
except Exception:
    res["scaffold_error"] = traceback.format_exc()[-3000:]
    save()
    log("SCAFFOLD CAPTURE FAILED:\n" + res["scaffold_error"])
    raise

# ---- detach transformer + VAE to GPU (precompute-embeds training footprint) ----
from accelerate.hooks import remove_hook_from_module  # noqa: E402

remove_hook_from_module(tf, recurse=True)
remove_hook_from_module(vae, recurse=True)
torch.cuda.empty_cache()
tf.to("cuda"); vae.to("cuda")
if hasattr(tf, "enable_gradient_checkpointing"):
    tf.enable_gradient_checkpointing()
DEV = "cuda"

# ---- latent normalization (per-channel mean/std + scaling) ----
LC = 128
vc = vae.config
mean = getattr(vc, "latents_mean", None)
std = getattr(vc, "latents_std", None)
sf = float(getattr(vc, "scaling_factor", 1.0))
res["vae_has_mean_std"] = mean is not None and std is not None
res["vae_scaling_factor"] = sf
if mean is not None and std is not None:
    mean = torch.tensor(mean, device=DEV, dtype=torch.float32).view(1, LC, 1, 1, 1)
    std = torch.tensor(std, device=DEV, dtype=torch.float32).view(1, LC, 1, 1, 1)
else:
    mean = torch.zeros(1, LC, 1, 1, 1, device=DEV)
    std = torch.ones(1, LC, 1, 1, 1, device=DEV)


def normalize(z):      # raw VAE latent -> transformer space
    return (z.float() - mean) * sf / std


def denormalize(z):    # transformer space -> raw VAE latent
    return z.float() * std / sf + mean


def encode_clip(frames_list):
    """list of WIN uint8 HWC -> normalized latent (1,128,7,16,16)."""
    arr = np.stack(frames_list).astype(np.float32)              # (WIN,H,W,3)
    x = torch.from_numpy(arr).permute(3, 0, 1, 2)[None] / 127.5 - 1.0  # (1,3,WIN,H,W)
    x = x.to(DEV, torch.bfloat16)
    with torch.no_grad():
        enc = vae.encode(x)
        z = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else (
            enc.latent if hasattr(enc, "latent") else enc[0])
    return normalize(z)


def decode_mid(zt_norm, mid):
    """normalized latent -> the mid pixel frame as uint8 HWC."""
    z = denormalize(zt_norm).to(torch.bfloat16)
    with torch.no_grad():
        dec = vae.decode(z)
        v = dec.sample if hasattr(dec, "sample") else dec[0]    # (1,3,F,H,W) in [-1,1]
    v = ((v[0, :, mid].float().clamp(-1, 1) + 1) * 127.5).round().byte()  # (3,H,W)
    return v.permute(1, 2, 0).cpu().numpy()


def to_tokens(z):      # (1,128,7,16,16) -> (1,1792,128)
    return z.permute(0, 2, 3, 4, 1).reshape(1, -1, LC)


def to_latent(tok, t=7):   # (1,N,128) -> (1,128,t,16,16)
    return tok.reshape(1, t, 16, 16, LC).permute(0, 4, 1, 2, 3)


# ---- B. VAE identity round-trip (does the VAE preserve flat-colour identities?) ----
try:
    rt = []
    for c in clips:
        z = encode_clip(c["idmap"])
        gen_mid = decode_mid(z, c["mid"])
        pb = idmaps_to_boxes([gen_mid], palette=palette, min_area=200)
        rt.append({"iou_vs_gt": round(mean_iou(pb, c["gt"]) or 0.0, 3),
                   "pred_tracks": len({b.track_id for b in pb})})
    res["vae_roundtrip"] = rt
    save()
    log(f"VAE identity round-trip: {rt}")
except Exception:
    res["vae_roundtrip_error"] = traceback.format_exc()[-3000:]
    save()

# precompute clip latents (reused across modes)
rgb_lat = [encode_clip(c["rgb"]) for c in clips]      # (1,128,7,16,16) each
idm_lat = [encode_clip(c["idmap"]) for c in clips]
_r = res.get("t_over_sigma")
ratio = _r if isinstance(_r, (int, float)) and math.isfinite(_r) and _r > 0 else 1000.0
res["ratio_used"] = ratio
# latent-scale diagnostic: my encoded idmap latent vs the model's expected token scale (the
# captured noised hidden_states at sigma~1 is ~unit). Wildly different => a missing normalization.
res["latent_diag"] = {
    "idmap_lat_std": round(float(idm_lat[0].float().std()), 3),
    "idmap_lat_absmean": round(float(idm_lat[0].float().abs().mean()), 3),
    "scaffold_hidden_std": round(float(sc7["hidden_states"].float().std()), 3),
}
save()
log(f"latent_diag {res['latent_diag']}")


def build_kwargs(scaffold, hidden, timestep_per_token, sigma_scalar):
    kk = dict(scaffold)
    kk["hidden_states"] = hidden.to(torch.bfloat16)
    kk["timestep"] = timestep_per_token.to(scaffold["timestep"].dtype)
    if "sigma" in kk and torch.is_tensor(kk["sigma"]):
        kk["sigma"] = torch.full_like(kk["sigma"], float(sigma_scalar))
    return kk


def forward_velocity(scaffold, hidden, t_tokens, sigma, n_target_tokens):
    """Run tf, return predicted velocity for the LAST n_target_tokens tokens."""
    kk = build_kwargs(scaffold, hidden, t_tokens, sigma)
    o = tf(**kk)
    pred = o.sample if hasattr(o, "sample") else (o[0] if isinstance(o, (tuple, list)) else o)
    if pred.dim() == 5:                      # (B,C,t,h,w) -> (B, t*h*w, C) token layout
        pred = pred.permute(0, 2, 3, 4, 1).reshape(pred.shape[0], -1, pred.shape[1])
    return pred[:, -n_target_tokens:]


def run_mode(mode):
    """Overfit the two clips under `mode`, then reconstruct each from shared noise -> self/cross."""
    from peft import LoraConfig
    # fresh VIDEO-only attn q/k/v LoRA per mode (exclude the audio branch — paths from Stage A)
    vtargets = [n for n, m in tf.named_modules()
                if isinstance(m, torch.nn.Linear) and "audio" not in n
                and n.split(".")[-1] in ("to_q", "to_k", "to_v")
                and (".attn1." in n or ".attn2." in n)]
    adapter = f"sanity_{mode}"
    tf.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=vtargets,
                              lora_dropout=0.0), adapter_name=adapter)
    tf.set_adapter(adapter)
    trainable = [p for p in tf.parameters() if p.requires_grad]
    extra = []

    if mode == "channel_concat":
        scaffold, n_tok, n_lat = sc7, 1792, 7
        # expand proj_in 128 -> 256 (+128 zero-init), make it trainable
        pin = getattr(tf, "proj_in", None)
        if not isinstance(pin, torch.nn.Linear):
            pin = next(m for n, m in tf.named_modules()
                       if n.endswith("proj_in") and isinstance(m, torch.nn.Linear))
        old = pin.weight.data                                  # (inner, 128)
        new = torch.zeros(old.shape[0], 2 * LC, device=old.device, dtype=old.dtype)
        new[:, :LC] = old
        pin.weight = torch.nn.Parameter(new)
        pin.in_features = 2 * LC
        pin.weight.requires_grad_(True)
        if pin.bias is not None:
            pin.bias.requires_grad_(True); extra.append(pin.bias)
        extra.append(pin.weight)
    else:  # temporal_pack
        scaffold, n_tok, n_lat = sc14, 3584, 7   # 14 latent frames total; target = last 7

    opt = torch.optim.AdamW(trainable + extra, lr=LR)
    tf.train()
    losses = []
    for step in range(STEPS):
        i = step % 2
        x0 = to_tokens(idm_lat[i]).float()                    # (1,1792,128) target
        rgb = to_tokens(rgb_lat[i]).float()                   # (1,1792,128) condition
        sigma = float(torch.rand(1))
        noise = torch.randn_like(x0)
        xt = (1 - sigma) * x0 + sigma * noise                 # noised target
        vtgt = noise - x0
        if mode == "channel_concat":
            hidden = torch.cat([rgb, xt], dim=-1)             # (1,1792,256)
            t_tok = torch.full((1, n_tok), sigma * ratio, device=DEV)
        else:
            hidden = torch.cat([rgb, xt], dim=1)              # (1,3584,128) [cond ; target]
            t_tok = torch.cat([torch.zeros(1, n_tok // 2, device=DEV),
                               torch.full((1, n_tok // 2), sigma * ratio, device=DEV)], dim=1)
        v = forward_velocity(scaffold, hidden, t_tok, sigma, 1792)
        loss = F.mse_loss(v.float(), vtgt)
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(float(loss))
        if step % 25 == 0:
            log(f"[{mode}] step {step} loss {np.mean(losses[-25:]):.4f}")

    # reconstruct each clip from the SAME noise, conditioned on its own RGB (few-step Euler)
    tf.eval()
    gens = []
    fixed_noise = torch.randn(1, 1792, LC, device=DEV)
    sig = torch.linspace(1.0, 0.0, 31, device=DEV)   # 30 Euler steps (5 was too crude → blurry,
                                                     # incomplete maps; v4 pred_tracks 2/4 vs gt 7/6)
    for i in range(2):
        rgb = to_tokens(rgb_lat[i]).float()
        x = fixed_noise.clone()
        with torch.no_grad():
            for k in range(len(sig) - 1):
                s, s_next = float(sig[k]), float(sig[k + 1])
                if mode == "channel_concat":
                    hidden = torch.cat([rgb, x], dim=-1)
                    t_tok = torch.full((1, n_tok), s * ratio, device=DEV)
                else:
                    hidden = torch.cat([rgb, x], dim=1)
                    t_tok = torch.cat([torch.zeros(1, n_tok // 2, device=DEV),
                                       torch.full((1, n_tok // 2), s * ratio, device=DEV)], dim=1)
                v = forward_velocity(scaffold, hidden, t_tok, s, 1792)
                x = x + (s_next - s) * v.float()              # flow Euler 1->0
        gen_mid = decode_mid(to_latent(x, n_lat), clips[i]["mid"])
        gens.append(gen_mid)

    out = {"final_loss": round(float(np.mean(losses[-10:])), 4),
           "loss_curve": [round(float(np.mean(losses[max(0, j - 10):j + 1])), 4)
                          for j in (0, STEPS // 4, STEPS // 2, 3 * STEPS // 4, STEPS - 1)],
           "trainable_params": int(sum(p.numel() for p in (trainable + extra)))}
    pb = [idmaps_to_boxes([g], palette=palette, min_area=200) for g in gens]
    out["self"] = [round(mean_iou(pb[i], clips[i]["gt"]) or 0.0, 3) for i in range(2)]
    out["cross"] = [round(mean_iou(pb[0], clips[1]["gt"]) or 0.0, 3),
                    round(mean_iou(pb[1], clips[0]["gt"]) or 0.0, 3)]
    out["pred_tracks"] = [len({b.track_id for b in pb[i]}) for i in range(2)]
    out["self_minus_cross"] = round((np.mean(out["self"]) - np.mean(out["cross"])), 3)
    # clean up adapter + proj_in for the next mode
    tf.delete_adapter(adapter) if hasattr(tf, "delete_adapter") else None
    return out


# ---- C. overfit + decision. temporal_pack first (proven Cosmos-i2v analog, no proj_in
# surgery); channel_concat is deferred — it needs trainable proj_in expansion and didn't
# converge in the combined run, and running it in the same process mutates the model and
# breaks the next mode's LoRA targeting. Validate the proven mode cleanly first. ----
for mode in ("temporal_pack",):
    try:
        torch.cuda.reset_peak_memory_stats()
        res[mode] = run_mode(mode)
        res[mode]["peak_vram_gb"] = vram_peak()
        save()
        log(f"{mode}: self {res[mode]['self']} cross {res[mode]['cross']} "
            f"margin {res[mode]['self_minus_cross']} peak {res[mode]['peak_vram_gb']}")
    except Exception:
        res[f"{mode}_error"] = traceback.format_exc()[-3000:]
        save()
        log(f"{mode} FAILED:\n" + res[f'{mode}_error'])

# verdict: conditioning WORKS for a mode iff mean(self) clearly exceeds mean(cross) AND self is
# meaningfully above the ~0.07 no-signal floor — not merely "the only mode that ran".
def winner():
    best, best_margin = None, 0.10   # require a clear positive margin to declare a pass
    for m in ("channel_concat", "temporal_pack"):
        r = res.get(m)
        if r and "self_minus_cross" in r and r["self_minus_cross"] > best_margin \
                and np.mean(r.get("self", [0])) > 0.3:
            best, best_margin = m, r["self_minus_cross"]
    return best


res["winner"] = winner()
res["best_margin"] = {m: res[m].get("self_minus_cross") for m in ("channel_concat", "temporal_pack")
                      if m in res and "self_minus_cross" in res[m]}
res["status"] = "DONE"
save()
log(f"LTX-COND-SANITY-DONE winner={res['winner']} -> {RES}/sanity.json")
