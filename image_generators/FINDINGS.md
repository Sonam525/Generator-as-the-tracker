# FINDINGS — Generative Tracking via Identity-as-Persistent-Color

Consolidated as of 2026-06-18. This is the narrative synthesis; the private experiment log
holds the full per-run log (config, seed, cost, gate verdict). Priority order throughout:
**identity > mask > detection > throughput** (throughput never claimed).

---

## TL;DR

- **Perception-as-generation works.** A LoRA-tuned Cosmos3-Nano, conditioned on the frame
  via the i2v spatial path and trained on SAM-3.1 silhouette labels, emits clean instance
  ID-maps: DanceTrack val **mIoU 0.585, det@0.5 0.625**.
- **The headline bet — making the *generator paint* persistent identity — fails cleanly.**
  Three injection mechanisms (temporal-pixel, text-slot, cross-attention) do not bind
  identity; root cause is a weak colour-binding gradient under the velocity-MSE objective.
- **A generative perceiver + a learned re-id bank solves association, drift-free.** On
  DanceTrack it beats the no-state heuristic by **IDF1 +40%, ID-switches −66%**, with
  detection unchanged (no autoregressive drift).
- **Re-acquisition across occlusion remains unsolved** — 0 on DanceTrack (clean) and 0 on
  zero-shot MOT17 (confounded by detection). An appearance diagnostic shows the ceiling:
  even DINOv3 re-identifies the right track only ~35 % of the time on these datasets.

Net: the *original* contribution (generator paints identity) is a clean negative; a
*different*, working contribution emerged (generative detector + learned re-id for
identity-as-colour). Re-acquisition is the open hard problem, consistent with the field.

---

## The arc, gate by gate

| Phase | Result | Verdict |
|---|---|---|
| **Codec / data** | round-trip IoU ≥ 0.99 (K=48 CIELAB palette); DanceTrack 40 train vids; SAM-3.1 pseudo-masks 41,796 frames @ **97 % box→silhouette** coverage | foundation green |
| **P-A** (NB2, zero-training) | identity-as-colour holds frame-to-frame; long-gap re-id fails zero-shot | hypothesis viable; state problem real |
| **P0** (perception fine-tune) | filled-box+unconditioned plateaued mIoU ~0.19 → **traced to a conditioning bug** → i2v + SAM silhouettes → **mIoU 0.585 / det@0.5 0.625** | GATE-0 cleared |
| **Gate-2B floor** (no-state) | conditioned perception + heuristic (IoU/colour) association: recall@0.5 ~0.74, **IDF1@0.3 ~0.28, IDSW ~30, reacq 0** | the floor to beat |
| **P2 generation-conditioning** | temporal (drifts), text-slot bank (ignored), cross-attn bank (ignored) | **clean negative** (×3) |
| **P2 re-id pivot** | perceiver + `IdentityBank` (appearance+motion): **IDF1 +40 %, IDSW −66 %**, drift-free; reacq still 0 | **association win** |
| **Re-acq diagnostic** | appearance discriminability (GT crops): VAE top1 0.17 (≈chance), DINOv3 0.35 (~2× chance), negative margin | appearance-bounded |
| **MOT17** | data ingested (direct-parallel HTTP); zero-shot perception sparse-only; re-id reacq 0 (detection-confounded); cheap de-confound did not help | needs perception fine-tune for a clean test |

---

## Key numbers

**P0 perception (DanceTrack internal-val):**

| run | val mIoU | det@0.5 |
|---|---|---|
| filled-box, unconditioned (void) | ~0.19 | ~0.05 |
| **i2v-conditioned + SAM silhouettes** | **0.585** | **0.625** |

**Identity, DanceTrack 3 occlusion clips (chained, @IoU 0.3 unless noted):**

| | recall@0.5 | IDF1@0.3 | IDSW@0.5 | reacq |
|---|---|---|---|---|
| No-state floor | 0.74 | 0.28 | ~30 | 0 |
| **Re-id (VAE appearance)** | 0.74 | **0.39** | **~10** | 0 |
| Re-id (DINOv3 appearance) | 0.74 | 0.38 | ~9 | 0 |

**MOT17 zero-shot re-id (4 sparse val-halves, 41 occlusion events):** IDF1@0.3 0.21–0.37;
**reacq 0** — but detection-confounded (zero-shot recall@0.3 0.22–0.58; over-fragmentation
from unstable masks). A cheap motion-primary de-confound did not help → detection, not
matching, is the binding limit; a clean test needs MOT17 perception fine-tuning.

---

## What works (keep)

1. **i2v spatial conditioning** for perception-as-generation: pack the RGB frame as the
   clean condition latent (frame 0) and the ID-map as the denoised target (frame 1),
   `num_frames=5` → 2 latent frames. Validated by an overfit sanity (self-IoU 0.76–0.85 ≫
   cross). Lives in `scripts/p0_finetune.py`; the adapter is `models/cosmos.py`.
2. **The conditioned perceiver** checkpoint `checkpoints/p0_stage1_sam/lora_best`.
3. **The re-id bank** `models/identity_bank.py` (per-track embedding, momentum-update when
   visible, **freeze under occlusion**, motion, appearance+motion `assign`) — unit-tested,
   backbone-agnostic; the drift-free way to carry identity on this stack.
4. **SAM-3.1 Mode-1 pseudo-masks** (`data/pseudo_masks.py`) and the eval harness
   (`eval/mot_metrics.py` incl. detection-recall + reacquisition, `eval/chained_eval.py`).

## Clean negatives (report)

1. **`num_frames=1` ignores the conditioning image** in `Cosmos3OmniPipeline` — every
   pre-fix P0/P1/P2 run was unconditioned (text-only). The single most important bug; its
   discovery turned a "weak localization ceiling" into a solved perception problem.
2. **The generator will not paint bank identity via LoRA-scale conditioning.** Text-slot and
   cross-attention injections both yield colour-binding 0.0 even with an easy positional cue
   and 800 steps. Likely cause: colour is a small fraction of the 48-ch latent variance, so
   velocity-MSE gives identity-routing too weak a gradient.
3. **Autoregressive generation-conditioning drifts** (temporal arm): cuts switches but
   collapses detection on its own outputs (exposure bias + frame dimming).
4. **Re-acquisition is appearance-bounded** on DanceTrack (uniform costumes by design) and
   unproven-but-0 on zero-shot MOT17.

---

## Root-cause notes (the two pivotal discoveries)

- **Conditioning:** the frame must enter as a *spatial* latent (i2v condition frame), not
  via the understanding/text path. Text-stream conditioning does not drive spatial output.
- **Identity placement:** put identity where the gradient is strong — a *directly-supervised
  association* (the re-id bank on stable per-frame perception) — not buried in the
  generator's velocity loss. Anything that chains the generator on its own outputs drifts;
  fresh per-frame perception + external association is drift-free.

---

## If resumed — next steps (in priority order)

1. **MOT17 perception fine-tune** (the one clean test we didn't run): SAM masks on the
   train-halves → fine-tune P0 on MOT17 → re-id eval on val-halves. The *only* way to get an
   unconfounded MOT17 re-acquisition number; the appearance ceiling suggests it stays modest.
2. **Re-acquisition via motion/trajectory**, not appearance (appearance is bounded): a
   learned motion model in the bank instead of linear extrapolation.
3. **Per-encoder matching calibration** for the bank (the DINOv3 cosine scale ≠ VAE's; the
   47-colour-cap over-fragmentation showed `match_thr` must be calibrated, not reused).
4. Optional: HOTA/IDF1 via TrackEval on full DanceTrack val for an anchor-comparable P4
   number, once re-acquisition is addressed.

## Cost

NB2 API: **$20.85 / $30** prepaid (unchanged since P-A). GPU: ~40 A100-hr across the project
(DanceTrack DL + Cosmos calibration + SAM pass ~8.5h + P0/P1/P2 train+eval cycles + MOT17
ingestion + evals). Cluster stopped.

## Kill-switch status

Not triggered. P-A worked; perception works; the re-id pivot beats the no-state floor on
association. The contribution shifted from "generator paints identity" (clean negative) to
"generative detector + learned re-id for identity-as-colour" (works for association;
re-acquisition open). This is a publishable result with clear positive and negative findings,
not a dead end.
