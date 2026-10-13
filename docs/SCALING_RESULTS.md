# Data Scaling — Does More Data Let Alternatives Overtake the MLP?

> Hypothesis under test: "With more data, spatial-mixing projectors (self_attn, honeybee)
> catch up to or overtake the pretrained MLP." Source: `scaling_result.json`.
> Figures: `figs/fig7_scale_effect.png` (500 vs 30k bars), `figs/fig8_scale_line.png` (lines),
> `figs/fig5_scaling_curve.png`.
> Setting: lr 2e-5, steps = 2·n_train, val 256, jaco_play.
> **Why lr 2e-5 here (vs 2e-4 in the main comparison):** a single common lr is needed so the
> LayerNorm-less variants (mlp_scratch, maxinfo_scratch) don't diverge (they collapse at
> 2e-4). This conservative lr under-serves honeybee: at its own lr **2e-4** honeybee reaches
> **0.0477** (main table), but here it shows 0.073–0.082 — a learning-rate artifact, not a
> contradiction. Even at 0.0477 it still does not beat the frozen MLP (0.0397).

## Action L1 by training-set size
| variant | 500 | 2,000 | 5,000 | 10,000 | 30,000 | trend |
|---|---|---|---|---|---|---|
| **baseline_mlp_frozen** | 0.0397 | 0.0397 | 0.0397 | 0.0397 | 0.0397 | flat (no training) |
| self_attn | 0.0983 | 0.0817 | 0.0829 | 0.0747 | **0.0728** | improves, plateaus |
| honeybee | 0.0894 | 0.0876 | 0.0730 | 0.0753 | 0.0815 | noisy, ~flat |
| mlp_scratch | 0.0719 | 0.1048 | 0.0758 | 0.0808 | 0.0901 | no clear gain |

## Reading the curve
- **self_attn** benefits most from scale: 0.098 → 0.073 from 500 → 30k. But it **plateaus
  around 0.073, still ~1.8× the frozen MLP's 0.0397**.
- **honeybee / mlp_scratch** show no reliable improvement with scale (within noise).
- The frozen MLP is data-independent (not trained) and stays best across the whole range.

## Verdict
The "more data flips the result" hypothesis is **not supported up to 30k samples**. Trained-
from-scratch projectors narrow the gap but do not close it; the **pretrained MLP remains the
accuracy ceiling**. This is consistent with the disentangle finding that real visual
grounding comes from large-scale pretraining (Exp2), which a 30k-sample fine-tune cannot
reproduce.

## maxinfo_scratch (in progress)
maxinfo_scratch is being added to `figs/fig7_scale_effect.png`. Note: at @500 with only
1,000 optimization steps the 185M, LayerNorm-less model under-converges (~0.19); the @30,000
cell (60,000 steps) is the fair point and is running. This will be folded in when complete.

## Connection to the main conclusion
Scaling reinforces the two-sided result: **same-input** alternatives **don't** win on
spatial/accuracy even with 60× more data, while their token-compression efficiency advantage
(INFERENCE_EFFICIENCY.md) is independent of data size. **Compress for deployment; keep the MLP
for spatial fidelity.**

> **⚠ Follow-up (PAPER.md §6).** These scaling runs vary data size but keep the projector's
> **input fixed** (penultimate ViT layer). The follow-up changes the *input* instead — feeding
> **multi-scale ViT features** — and there the spatial-information transfer **does** scale up:
> +14.8% → +30.8% (token-wise) as data grows 1.5k → 6k, and +34% with cross-scale attention
> (all p<0.02, all seeds positive). So "more data doesn't help" holds for *same-input*
> variants; widening the *input* is what moves spatial-information transfer.
