"""Write the SportsMOT training configs from the DanceTrack templates (no recipe changes).

Usage:
  python video_generator/make_sportsmot_configs.py --val-vids v_a,v_b,...   # 8 fixed val videos (pre-registered rule)
  python video_generator/make_sportsmot_configs.py --val-vids ... --phase-a-step 1500   # also write phase 3 (B)

Differences from the DanceTrack configs are ONLY: dataset paths, the 8-video validation subset,
validation/checkpoint interval 750 (instead of 500) to hold the in-loop validation budget, output dirs
and tags. Everything else (LoRA targets, rank, lr, steps, prefix condition p=0.5 / 2 latent frames,
grad-accum 4 in phase B, sampler settings) is copied verbatim.
"""
import argparse
import copy
import io
import os

import yaml

VOL = os.environ.get("GENTRACK_VOL", "/Volumes/cornell_catalog/cornell_schema/blob_video/Generator is the tracker")
ap = argparse.ArgumentParser()
ap.add_argument("--val-vids", required=True, help="comma-separated 8 SportsMOT val video names")
ap.add_argument("--phase-a-step", type=int, default=None, help="selected phase-A checkpoint step (writes phase 3)")
ap.add_argument("--interval", type=int, default=750)
args = ap.parse_args()
vids = [v.strip() for v in args.val_vids.split(",") if v.strip()]
assert len(vids) == 8, vids


def load(p):
    return yaml.safe_load(io.open(p, encoding="utf-8"))


def dump(cfg, p, header):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print("wrote", p)


def retarget(cfg, phase):
    cfg = copy.deepcopy(cfg)
    cfg["data"]["preprocessed_data_root"] = f"{VOL}/ltx_ds/sportsmot_phase2_train/.precomputed"
    tmpl = cfg["validation"]["samples"][0]
    new = []
    for v in vids:
        e = copy.deepcopy(tmpl)
        for c in e["conditions"]:
            if c.get("type") == "reference":
                c["video"] = f"{VOL}/ltx_ds/sportsmot_phase2_val/rgb/{v}_000001_window.mkv"
        new.append(e)
    cfg["validation"]["samples"] = new
    cfg["validation"]["interval"] = args.interval
    cfg["checkpoints"]["interval"] = args.interval
    cfg["output_dir"] = f"/local_disk0/ltx_out/sm_{phase}"
    cfg["wandb"]["tags"] = ["ltx2", "ic-lora", "sportsmot", phase]
    return cfg


a = retarget(load("video_generator/configs/ltx_v2v_ic_lora_phase2.yaml"), "phase2")
dump(a, "video_generator/configs/ltx_v2v_ic_lora_sm_phase2.yaml",
     "# SportsMOT phase A (translation): the DanceTrack phase-2 recipe verbatim, retargeted to the\n"
     "# SportsMOT train clips; in-loop validation on a FIXED 8-video val subset every 750 steps.\n"
     f"# val subset (pre-registered rule, 3 basketball + 3 football + 2 volleyball): {', '.join(vids)}\n")

if args.phase_a_step is not None:
    b = retarget(load("video_generator/configs/ltx_v2v_ic_lora_phase3.yaml"), "phase3")
    b["model"]["load_checkpoint"] = (f"{VOL}/checkpoints/ltx_v2v_ic_lora_sm_phase2/checkpoints/"
                                    f"lora_weights_step_{args.phase_a_step:05d}.safetensors")
    dump(b, "video_generator/configs/ltx_v2v_ic_lora_sm_phase3.yaml",
         "# SportsMOT phase B (continuation): the DanceTrack phase-3 recipe verbatim (prefix p=0.5, 2 latent\n"
         f"# frames, grad-accum 4), resuming the phase-A checkpoint selected by val fg-IoU (step {args.phase_a_step}).\n")
