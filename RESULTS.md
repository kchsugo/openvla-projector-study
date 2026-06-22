# Results — All Numbers

> jaco_play (Open-X) · train 1,500 / val 256 · seed 0 · **3,000 steps**, lr 2e-4, batch 1,
> 4-bit · RTX 5060 8GB · projector-only training (LLM + vision frozen).
> Source: `compare_real_result.json`, `compare_lowlr_diag.json`, `bench_inference_result.json`,
> `scaling_result.json`, `vision_dep_result.json`, `spatial_probe_result.json`.

## 1. Main comparison (9 variants)
| variant | Action L1 ↓ | Action MSE ↓ | token_acc ↑ | tokens | params | note |
|---|---|---|---|---|---|---|
| **baseline_mlp_frozen** | **0.0397** | 0.0247 | 0.746 | 256 | 0 | pretrained, best L1 |
| baseline_mlp_trained | 0.0472 | 0.0270 | 0.779 | 256 | 71.4M | |
| maxinfo | 0.0397 | 0.0247 | 0.746 | 256 | 114.1M | γ≡0 = **deadlock**, never trains (§6.1, PAPER.md) |
| honeybee | 0.0477 | **0.0213** | 0.759 | 64 | 24.3M | best MSE, −75% tokens |
| mlp_scratch_ln | 0.0481 | 0.0231 | 0.766 | 256 | 71.4M | LN-only |
| self_attn | 0.0534 | 0.0275 | 0.762 | 256 | 19.0M | spatial mixing |
| cross_attn | 0.0669 | 0.0438 | 0.757 | 64 | 19.1M | −75% tokens |
| mlp_scratch | 0.2059 | 0.1305 | 0.000 | 256 | 71.4M | **collapse (no LN)** |
| maxinfo_scratch | 0.1944 | 0.1217 | 0.060 | 256 | 185.5M | **collapse (no LN)** |

### L1 vs MSE — different rankings (report both)
- **L1 (mean abs error):** frozen is best (0.0397).
- **MSE (squared, outlier-sensitive):** honeybee (0.0213) and mlp_scratch_ln (0.0231) edge
  out frozen (0.0247) — meaning they make fewer *large* errors, but their average error is
  still higher. This MSE edge appears **with and without** spatial mixing, so it is a
  LayerNorm + in-distribution-fit effect, not a mixing effect (see DISENTANGLE_RESULTS.md).

## 2. Stability diagnostic (no-LayerNorm variants)
At lr 2e-4 the LayerNorm-less variants diverge; at lr 2e-5 they recover:
| variant | lr 2e-4 (L1) | lr 2e-5 (L1) | lr 2e-5 (MSE) | acc |
|---|---|---|---|---|
| mlp_scratch | 0.2059 (collapse) | 0.071 | 0.0423 | 0.741 |
| maxinfo_scratch | 0.1944 (collapse) | 0.0474 | 0.0173 | 0.752 |

→ The collapse is an **optimization/normalization** issue, not a capacity or spatial-info
issue. Even recovered, neither beats the frozen MLP on L1.

## 3. On-device inference (measured, 4-bit, greedy 7-token action)
| variant | tokens | latency (ms) | std | peak VRAM |
|---|---|---|---|---|
| baseline_mlp_frozen | 256 | 331.0 | 3.0 | 4.64 GB |
| honeybee | 64 | 259.7 | 1.3 | 4.62 GB |
| cross_attn | 64 | 260.5 | 1.7 | 4.60 GB |

−75% visual tokens → −21% latency, negligible VRAM change. See INFERENCE_EFFICIENCY.md.

## 4. Data scaling — Action L1 (lr 2e-5, steps = 2·n)
| variant | 500 | 2000 | 5000 | 10000 | 30000 |
|---|---|---|---|---|---|
| baseline_mlp_frozen | 0.0397 | 0.0397 | 0.0397 | 0.0397 | 0.0397 |
| self_attn | 0.0983 | 0.0817 | 0.0829 | 0.0747 | 0.0728 |
| honeybee | 0.0894 | 0.0876 | 0.0730 | 0.0753 | 0.0815 |
| mlp_scratch | 0.0719 | 0.1048 | 0.0758 | 0.0808 | 0.0901 |

No alternative crosses below frozen 0.0397 even at 30k. See SCALING_RESULTS.md.
**lr note:** this sweep uses lr **2e-5** (common lr so LN-less variants don't diverge), which
under-serves honeybee — at its own lr **2e-4** honeybee reaches **0.0477** (§1 main table),
not 0.073–0.082. Same model, different lr; conclusion (doesn't beat frozen 0.0397) unchanged.
(maxinfo_scratch scaling is being added; @500/1000-step under-converges to ~0.19.)

## 5. Disentangle probes
| variant | vision_shift (Exp2) ↑ | action readout R² (Exp3) ↑ | token_var |
|---|---|---|---|
| baseline_mlp_frozen | **0.128** | −6.82 | 0.20 |
| maxinfo | 0.128 | −6.82 | 0.20 |
| honeybee | 0.094 | −2.15 | 0.29 |
| mlp_scratch | 0.069 | NaN (collapse) | NaN |
| self_attn | 0.005 | 0.28 | 0.70 |
| mlp_scratch_ln | 0.002 | **0.42** | 8.60 |

Interpretation in DISENTANGLE_RESULTS.md. Short version: frozen is most **vision-grounded**
(Exp2); mlp_scratch_ln (no mixing) has the highest **action readout** (Exp3) — spatial
mixing helps neither.

## Headline
- **Efficiency: SUCCESS** — compressing projectors give −75% tokens / −21% latency for free.
- **Spatial information: the base MLP wins *among same-input variants*** — no variant that
  re-processes the penultimate-layer features transfers more; LayerNorm + pretraining are the
  levers, not spatial mixing.
- **Follow-up (PAPER.md §6):** once you change the **input** (feed multi-scale ViT layers, not
  just the penultimate), a projector **does** transfer significantly more spatial information
  to the LLM — **+31% token-wise (p=0.0018)**, **+34% cross-scale attention (p=0.019)**, all
  seeds positive. Also, the `maxinfo` "γ→0" above was a **gradient deadlock**, not evidence.
