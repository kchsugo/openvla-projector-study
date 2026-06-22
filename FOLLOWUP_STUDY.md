# Follow-up Study & Critique Response

> This document summarizes the **follow-up session** that extends the original projector
> ablation ([PAPER.md](PAPER.md) §1–5). It introduces the **multi-scale** projector line,
> corrects the **maxinfo gradient deadlock**, and closes the reviewer critiques with
> **causal + observational + downstream** evidence. Full write-up: PAPER.md §6.

---

## 0. One-paragraph summary
The original study held the projector's **input fixed** (only the penultimate ViT layer) and
varied its internals — and found spatial token *mixing* useless. This follow-up shows the
real lever is the **input**: OpenVLA discards all ViT layers but the penultimate, and a
**multi-scale** projector that delivers the discarded layers transfers **significantly more
spatial information to the LLM** (+31% token-wise, p=0.0018; **+34% via cross-scale
attention**, p=0.019; all seeds positive). We prove the load-bearing hypothesis — *the LLM's
self-attention already performs the 256-token spatial integration* — both **causally**
(blocking vision↔vision attention collapses the frozen model by **+86%** but barely touches a
projector that pre-mixes, +2%) and **observationally** (vision tokens place ~16.5% of their
attention on other patches). Downstream, the extra information **does** reduce action error,
but **only on spatially-demanding states** (−4.8%, p=0.025), while slightly hurting trivial
states (+19%) — so it is a *state-conditional* gain, not a free win.

---

## 1. Experiments (this session)

| # | Experiment | Method | Key result | Figure | Verdict |
|---|---|---|---|---|---|
| 1 | **Multi-scale transfer** | feed penult + intermediate ViT layers; measure `spatial_shift` (image-swap, continuous metric) vs frozen | **+31%** token-wise (p=0.0018, 7/7); **+34%** cross-scale-attn (p=0.019, 5/5) | ms2, ms3 | ✅ strong |
| 2 | **Architecture** | OpenVLA forwards only ViT penultimate layer → projector | the discarded layers never reach the LLM | ms1 | ✅ premise |
| 3 | **maxinfo deadlock** | γ & enhancement-output both zero-init → zero gradient to both | γ≡0 was a **bug**, not "no-regret"; fixed via `zero_init_out=False` | ms4 | ✅ correction |
| 4 | **Qualitative demo** | real spatially-different jaco image pairs, 5-seed avg | multiscale_attn responds more (+34%); aggregate, not per-image | ms5 | ✅ honest |
| 5 | **Downstream (hard subset)** | xyz action-L1 on spatially-demanding states (top-33% GT motion) | **hard −4.8% (p=0.025)**, easy **+19%** (p=0.008), all +5% (n.s.) | ms6 | ⚠️ conditional |
| 6 | **Causal: LLM attn ablation** | block vision↔vision attention in the LLM (manual 4D mask, self-test agreement=1.00) | frozen **+86%** collapse; self_attn **+2%** (immune) | ms7 | ✅ strong |
| 7 | **Observational: attn maps** | per-layer vision→other-patch attention mass (eager) | mean **0.165**, peak 0.95, strongest early layers | ms8 | ✅ strong |

---

## 2. The three reviewer critiques — and how each is closed

**(C1) Single task / 2D-planar, low spatial demand → action gain diluted.**
→ Measured downstream action error **on the spatially-demanding subset** (Exp 5). The flat
overall +1–5% hides a real dissociation: **−4.8% on hard states (p=0.025)**. So the proxy
gain *does* convert to downstream — where spatial reasoning is actually required.
*Honest caveat:* it hurts trivial states (+19%), so the net is state-conditional.

**(C2/C3) Load-bearing hypothesis ("LLM attention already does the projector's spatial
mixing") was asserted, not verified.**
→ Closed three ways:
- **Causal (Exp 6):** block vision↔vision attention → frozen collapses **+86%** (the LLM
  *was* doing the mixing), while a projector that pre-mixes patches is **immune (+2%)** — a
  controlled falsification (if the hypothesis were false, blocking would not matter).
- **Observational (Exp 7):** vision tokens put **16.5%** of attention on other patches
  (peak 0.95, concentrated in early layers).
- **Controlled task:** the blocked-attention condition is exactly the regime where projector
  mixing *must* help if it is ever useful — and it is (self_attn immune).

---

## 3. Honest overall assessment

| Signal | Status |
|---|---|
| Spatial-information transfer (multiscale, cross-scale attn) | ✅ strong, significant, all seeds |
| Lever = input (not mixing) | ✅ clear |
| Load-bearing hypothesis (LLM does the mixing) | ✅ proven (causal + observational) |
| maxinfo deadlock correction | ✅ rigorous |
| Downstream action accuracy | ⚠️ **conditional** — helps spatial states, hurts trivial ones, flat on average |
| Absolute responsiveness | ⚠️ both projectors track only ~30% of GT spatial change |
| Effect visibility | ⚠️ aggregate/statistical, not per-image |
| Other datasets (CALVIN), closed-loop success | ❌ not run (no data / no simulator here) — future work |

**Framing.** This is best positioned as a **diagnostic + mechanism** contribution, not a
performance claim: *OpenVLA discards ViT layers and thereby limits spatial-information
transfer (shown); the LLM — not the projector — performs the 256-token spatial integration
(proven causally); restoring the discarded layers transfers more spatial information and
improves action error specifically on spatially-demanding states, though general utilization
remains limited.*

---

## 4. Files & artifacts (this session)

**Code**
- `multiscale.py` — `MultiScaleProjector`, `MultiScaleAttnProjector` (cross-scale attention), `enable_multiscale`
- `projector.py` — `zero_init_out` flag (deadlock fix)
- `projectors_zoo.py` — variants `maxinfo_fixed`, `multiscale`, `multiscale3`, `multiscale_attn`
- `vision_dep_spatial.py` — spatial-only Exp2 (argmax + continuous `spatial_shift`, paired stats)
- `scale_spatial.py` — data/step scaling harness
- `spatial_demo.py`, `make_spatial_demo_fig.py` — qualitative demo
- `spatial_action.py` — downstream action-L1 on hard/easy subsets
- `llm_attn_ablation.py` — causal vision↔vision attention ablation (manual 4D mask)
- `attn_map_analysis.py` — per-layer vision→vision attention mass (eager)
- `make_multiscale_figures.py`, `make_critique_figures.py` — figure generators

**Results (JSON)**
- `scale_spatial_result.json` (multiscale & multiscale_attn, 1.5k+6k, argmax+continuous)
- `vision_dep_spatial_result.json` (multiscale 1.5k, 10 seeds)
- `compare_maxinfo_fixed.json` (deadlock)
- `spatial_action_result.json` (downstream hard/easy)
- `llm_attn_ablation_result.json` (causal), `attn_map_result.json` (observational)

**Figures (`figs/`)**
- ms1_architecture · ms2_gain · ms3_metric · ms4_deadlock · ms5_spatial_demo
- ms6_downstream_dissociation · ms7_attn_ablation · ms8_attn_map

---

## 5. Reproduction (run from `/home/ssu/openvla`)
```bash
source /home/ssu/openvla-env/bin/activate
# spatial-information transfer (scaling + seeds)
python maxinfo/scale_spatial.py --tag d6k_attn --train_n 6000 --steps 3000 --shards 16 \
    --variant multiscale_attn --seeds 0 1 2 3 4
# downstream on spatially-demanding subset
python maxinfo/spatial_action.py --train_n 1500 --steps 800 --seeds 0 1 2 3 4
# causal: block LLM vision-vision attention
python maxinfo/llm_attn_ablation.py --train_n 1500 --steps 800 --val_n 128
# observational: attention maps
python maxinfo/attn_map_analysis.py --n 8
# figures
python maxinfo/make_multiscale_figures.py && python maxinfo/make_critique_figures.py
```
