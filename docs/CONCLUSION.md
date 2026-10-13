# Conclusion — Honest Synthesis

## The one-paragraph takeaway
We set out to test whether OpenVLA's simple 2-layer MLP projector bottlenecks the transfer
of visual/spatial information, and whether spatial-mixing projectors (attention/conv) do
better. The answer splits cleanly in two. **On the on-device efficiency axis the study is a
success**: a token-compressing projector (honeybee) removes 75% of visual tokens and 21% of
latency with no meaningful accuracy loss — even a slightly *lower* MSE than the frozen MLP.
**On the spatial-information axis the original MLP wins**: no alternative transfers
visual/spatial information better, and three controlled experiments show the apparent
"attention advantage" was really **LayerNorm + large-scale pretraining**, not spatial token
mixing.

> **⚠ Follow-up update — see [PAPER.md](PAPER.md) §6.** A later analysis refines this synthesis:
> (1) the `maxinfo` "γ→0" below was a **gradient deadlock** (double zero-init → zero gradient
> to both the gate and the enhancement; it never trains), **not** a learned "no-regret"
> outcome; (2) "the frozen MLP is the spatial-information ceiling" is **input-bounded** —
> OpenVLA forwards only the penultimate ViT layer, and a **multi-scale** projector that
> delivers the discarded layers transfers **significantly more spatial information** to the
> LLM (**+31% token-wise, p=0.0018; +34% cross-scale-attention, p=0.019; all seeds positive**).
> The lever is *which features reach the LLM*, not mixing. (Scope: information transfer, not
> action accuracy — the action-L1 gain stayed ~1%.)

## What we confirmed
1. **Efficiency is a free lunch.** honeybee: 256→64 tokens, 331→260 ms, MSE 0.0247→0.0213.
   Deploy a compressing projector when you need a cheaper VLA.
2. **The base MLP is the spatial-information ceiling.** Frozen MLP has the best Action L1
   (0.0397); it is the most visually grounded (vision_shift 0.128, highest); and 30k-sample
   scaling does not let any alternative overtake it.
3. **Spatial mixing is not the lever.** With LayerNorm held constant, adding attention/conv
   gives no accuracy gain (self_attn 0.0534 ≥ mlp_scratch_ln 0.0481) and no readout gain
   (R² 0.28 < 0.42). The collapse of the scratch MLP was a **normalization** failure, fixed
   by LayerNorm alone — not a spatial-information failure.
4. **Capacity is not the lever.** self_attn (19M) is not beaten by mlp_scratch_ln (71.4M).
   *(Originally we also cited maxinfo's gate "collapsing to γ=0"; §6.1 shows that was a
   gradient **deadlock**, not a capacity finding — see the follow-up note above.)*

## What we corrected (intellectual honesty)
- The MSE edge of honeybee/mlp_scratch_ln over frozen is **not** evidence for spatial
  mixing: the **no-mixing** LN variant wins equally, and a mixing variant (self_attn) loses
  to frozen. The edge is LayerNorm stability + in-distribution fitting to jaco.
- High linear readout (Exp3) does **not** mean better visual grounding: scratch_ln/self_attn
  have high readout but near-zero vision_shift — they fit action priors, not the image.

## Practical guidance
| Goal | Recommendation |
|---|---|
| Cheaper on-device inference | Swap in a **compressing projector (honeybee)**: −75% tokens, −21% latency |
| Best spatial/visual fidelity | **Keep the pretrained MLP** (frozen); it is the accuracy & grounding ceiling |
| Training a projector from scratch | **Add LayerNorm** — its absence, not missing spatial mixing, is what breaks training |
| Adding capacity/attention "to be safe" | Use a residual gate, but **zero-init only γ, not the enhancement output** (double zero-init deadlocks — §6.1) |
| Transferring *more* spatial info | Feed the projector **multi-scale ViT features** (not just the penultimate layer); fuse across scale with attention (§6) |

## Limitations
Single dataset (jaco_play) and seed; small MSE gaps may be within noise; the spatial probe
is a weak proxy (no object-position labels, GT-action readout); vision_shift is an indirect
grounding measure. Conclusions are within projector-only fine-tuning up to 30k samples.

## Bottom line
> **On-device: success (compression works). Spatial information: the original MLP is best.**
> The real levers are **LayerNorm** and **pretraining**, not spatial token mixing.
