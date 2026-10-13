# Disentangling Experiments — Is it Spatial Mixing, or LayerNorm?

> Goal: the alternative projectors differ from the MLP on **several** axes at once
> (LayerNorm, spatial token mixing, capacity, token count). To tell *which* axis matters
> for transferring visual/spatial information, we ran three controlled experiments.
> Figures: `figs/exp1_accuracy.png`, `figs/exp2_vision_dep.png`, `figs/exp3_spatial_probe.png`,
> `figs/exp23_combined.png`, `figs/fig11_ln_decomposition.png`.

## The three experiments
| | Experiment | What it does | What it measures |
|---|---|---|---|
| **Exp1** | Accuracy 2×2 control | Train LN✓/✗ × mixing✓/✗ at the same setting (lr 2e-4, 3000 steps) | Action L1 — which knob recovers the scratch-MLP collapse |
| **Exp2** | Vision-dependency | Swap the input image, measure prediction change (`vision_shift`) | How much the model actually **uses the image** (visual grounding) |
| **Exp3** | Spatial probe | Ridge readout from projector output → GT action (R²) + token variance | How much spatial/action info is **linearly accessible** |

## Exp1 — Action accuracy (LayerNorm × spatial mixing)
| | mixing ✗ | mixing ✓ |
|---|---|---|
| **LN ✗** | mlp_scratch **0.2059** (collapse) | maxinfo_scratch **0.1944** (collapse) |
| **LN ✓** | mlp_scratch_ln **0.0481** (recovered) | self_attn 0.0534 |

- No LayerNorm → **collapse regardless of spatial mixing** (maxinfo_scratch *has*
  self-attention and still collapses).
- Add LayerNorm → recovers **without any spatial mixing** (mlp_scratch_ln).
- Add spatial mixing on top of LayerNorm → **no gain** (self_attn 0.0534 ≥ scratch_ln 0.0481).
- **⇒ The scratch-MLP failure was a normalization problem, not a spatial-information problem.**

## Exp2 — Visual grounding (vision_shift, higher = more vision-dependent)
| frozen | maxinfo | honeybee | mlp_scratch | self_attn | mlp_scratch_ln |
|---|---|---|---|---|---|
| **0.128** | 0.128 | 0.094 | 0.069 | 0.005 | 0.002 |

- The **pretrained frozen MLP** reacts to image swaps the most → most visually grounded.
- From-scratch variants (incl. LN, attn) barely react → they lean on jaco's **action
  priors**, not the visual input.
- ⇒ Real visual grounding comes from **large-scale pretraining**, not projector structure.

## Exp3 — Spatial/action info readout (R², higher = more linearly accessible)
| mlp_scratch_ln | self_attn | honeybee | frozen | maxinfo | mlp_scratch |
|---|---|---|---|---|---|
| **0.42** | 0.28 | −2.15 | −6.82 | −6.82 | NaN (collapse) |

- `mlp_scratch_ln (no mixing) > self_attn (mixing)` → spatial mixing does **not** increase
  linearly-decodable action info; LayerNorm-driven feature normalization does.
- Weak proxy: jaco has **no object-position labels**, so readout targets GT action; frozen's
  negative R² partly reflects 256-token ridge over-fitting (read with token_var = 0.20).

## Combined (Exp2 × Exp3) — the dissociation
`figs/exp23_combined.png`: scratch_ln / self_attn sit at **high readout, near-zero
vision_shift** (carry action info but ignore the image → fitting priors), while
**frozen/maxinfo** sit at **high vision_shift** (truly grounded). "Carries action info" ≠
"uses the image."

## Conclusion of the disentangle study
1. **Spatial mixing is not the lever.** Neither accuracy (Exp1) nor info readout (Exp3)
   improves when you add attention/conv. The no-mixing LayerNorm MLP matches or beats the
   spatial-mixing variants.
2. **LayerNorm is the lever for stable optimization** of from-scratch projectors.
3. **Pretraining is the lever for visual grounding** — the frozen MLP wins Exp2 decisively.
4. Net: among projectors that **re-process the same penultimate-layer input**, the original
   MLP is best for spatial/visual information; the alternatives' only genuine advantage is
   **token compression for on-device efficiency** (see INFERENCE_EFFICIENCY.md).

> **⚠ Follow-up (PAPER.md §6).** This study varied the projector's *internals* (mixing, LN,
> capacity) but held the **input** fixed (penultimate ViT layer only). When the input is
> widened to **multi-scale ViT features**, a projector transfers **significantly more spatial
> information** to the LLM (**+31% token-wise p=0.0018; +34% cross-scale-attention p=0.019**).
> So "mixing isn't the lever" stands; the real lever is the **input**.

### Limitations
Single dataset/seed; readout is a weak proxy; vision_shift is an indirect grounding measure.
Exp2/Exp3 list identical maxinfo & frozen values because maxinfo's gate γ stayed 0 — but
§6.1 shows that γ≡0 was a **gradient deadlock** (double zero-init), not a learned collapse,
so the maxinfo row here reflects an **untrained** enhancement.
