# Inference Efficiency — Token Compression for On-Device VLA (the SUCCESS result)

> Measured on OpenVLA-7B (4-bit), real image, 7-token greedy action generation,
> 20 runs after 3 warmups. Source: `bench_inference_result.json`.
> Figures: `figs/fig6_inference_efficiency.png`, `figs/fig9_efficiency_map.png`.

## The idea
The projector decides **how many visual tokens** enter the 7B LLM. The original MLP keeps
all **256** tokens. Convolutional (honeybee) and cross-attention (cross_attn) projectors
compress to **64** tokens (−75%). Fewer tokens → shorter LLM context → faster, cheaper
inference. The question: does compression cost accuracy?

## Measured cost
| variant | tokens | latency (ms) | latency std | peak VRAM |
|---|---|---|---|---|
| baseline_mlp_frozen | 256 | 331.0 | 3.0 | 4.64 GB |
| **honeybee** | **64 (−75%)** | **259.7 (−21%)** | 1.3 | 4.62 GB |
| cross_attn | 64 (−75%) | 260.5 (−21%) | 1.7 | 4.60 GB |

## Accuracy vs efficiency (the trade-off)
| variant | tokens | latency | Action L1 | Action MSE |
|---|---|---|---|---|
| frozen MLP | 256 | 331 ms | 0.0397 | 0.0247 |
| honeybee | 64 | 260 ms | 0.0477 (+0.008) | **0.0213 (better!)** |
| cross_attn | 64 | 261 ms | 0.0669 (+0.027) | 0.0438 |

- **honeybee** is the sweet spot: 75% fewer tokens, 21% faster, **lower MSE** than frozen and
  only +0.008 L1. Essentially free compression.
- cross_attn compresses just as much but pays a real accuracy cost — not all compressors are
  equal; the conv-based honeybee is the one to use.

## Why it matters on-device
- **Latency**: 331 → 260 ms per action (~21%) compounds over long-horizon control loops.
- **Memory/compute**: the LLM attends over 192 fewer tokens every step; the saving grows
  with longer prompts and KV-cache.
- **Quality**: honeybee keeps action quality (even improves MSE), so the compression is not
  paid for in control accuracy.

## Conclusion
**On the efficiency axis, the study succeeds.** A token-compressing projector (honeybee)
delivers a 75% token / 21% latency reduction with no meaningful accuracy loss — a clear win
for deploying OpenVLA on resource-constrained hardware.

> Caveat for the bigger picture: this efficiency win is about **compression**, not about
> better spatial-information transfer. Among **same-input** projectors the original MLP is
> best on spatial/visual information (see DISENTANGLE_RESULTS.md).
>
> **⚠ Follow-up (PAPER.md §6):** if instead you widen the projector's **input** to multi-scale
> ViT features, it *does* transfer significantly more spatial information to the LLM
> (+31% token-wise p=0.0018; +34% cross-scale-attention p=0.019). Efficiency (compression) and
> richer spatial-information transfer are two separate, complementary levers.
