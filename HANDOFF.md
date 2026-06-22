# Handoff — Reproduction Guide

Self-contained context for continuing this study in a fresh session. Conclusions live in
[CONCLUSION.md](CONCLUSION.md) / [PAPER.md](PAPER.md); this file is the *how-to-run*.

## TL;DR of findings
- **On-device efficiency: SUCCESS** — honeybee compresses 256→64 tokens (−75%), 331→260 ms
  (−21%), no real accuracy loss.
- **Spatial information (same input): the base MLP wins** — no variant re-processing the
  penultimate-layer features beats the frozen MLP; the "attention advantage" is
  **LayerNorm + pretraining**, not spatial mixing.
- **Follow-up (PAPER.md §6):** the lever is the projector's **input**. OpenVLA forwards only
  the penultimate ViT layer; a **multi-scale** projector that delivers the discarded layers
  transfers **significantly more spatial information** to the LLM (+31% token-wise p=0.0018;
  **+34% cross-scale-attention p=0.019**, all seeds positive). Also: `maxinfo`'s "γ→0" was a
  **gradient deadlock** (double zero-init), now fixed via `zero_init_out=False`.

## Environment
- venv: `source /home/ssu/openvla-env/bin/activate` (always use the **absolute** path).
- `export BNB_CUDA_VERSION=130` before running (bitsandbytes).
- Model: OpenVLA-7B, 4-bit nf4, `device_map={"":0}`, RTX 5060 8GB.
- Never modify OpenVLA source — projector is swapped at runtime (`model.projector = proj`).

## Code map (`maxinfo/`)
| file | purpose |
|---|---|
| `projectors_zoo.py` | all projector builders (mlp_scratch, mlp_scratch_ln, honeybee, self_attn, cross_attn, maxinfo, maxinfo_scratch, **maxinfo_fixed, multiscale, multiscale3, multiscale_attn**) |
| `train_eval.py` | main harness: trains each variant, writes `compare_real_result.json` (now has `--out`, `--variants`) |
| `multiscale.py` | **[§6]** `MultiScaleProjector`, `MultiScaleAttnProjector`, `enable_multiscale` (backbone multi-layer patch) |
| `projector.py` | maxinfo residual gate; **`zero_init_out` flag (deadlock fix)** |
| `vision_dep_spatial.py` | **[§6]** spatial-only Exp2: argmax + continuous `spatial_shift`, paired stats → `vision_dep_spatial_result.json` |
| `scale_spatial.py` | **[§6]** data/step scaling harness → `scale_spatial_result.json` |
| `make_multiscale_figures.py` | **[§6]** figs `ms1_architecture`, `ms2_gain`, `ms3_metric`, `ms4_deadlock` |
| `data.py` | `load_jaco_subset` (Open-X / RLDS), action encode/normalize |
| `scaling_curve.py` | data-scaling grid (LazyExamples for RAM safety), `scaling_result.json` |
| `vision_dep_per_variant.py` | Exp2 vision-dependency → `vision_dep_result.json` |
| `spatial_probe.py` | Exp3 spatial probe → `spatial_probe_result.json` |
| `bench_inference.py` | latency/VRAM benchmark → `bench_inference_result.json` |
| `run_disentangle.sh` | chains Exp1→Exp2→Exp3, writes `DISENTANGLE_DONE` |
| `make_*figure*.py` | all figure generators (see below) |

## How to reproduce
```bash
source /home/ssu/openvla-env/bin/activate
export BNB_CUDA_VERSION=130
# main 9-variant comparison
python maxinfo/train_eval.py
# disentangle chain (Exp1/2/3)
bash maxinfo/run_disentangle.sh
# data scaling
python maxinfo/scaling_curve.py --variants baseline_mlp_frozen mlp_scratch self_attn honeybee \
    --sizes 500 2000 5000 10000 30000 --lr 2e-5
# inference benchmark
python maxinfo/bench_inference.py
```

## Figure generators → outputs (`figs/`)
| script | output |
|---|---|
| `make_figures.py` | fig1_action_l1, fig2_acc_vs_l1, fig3_params_vs_l1, fig4_loss_curves |
| `make_ln_decomposition_figure.py` | fig11_ln_decomposition (LN×mixing 2×2) |
| `make_three_experiments_figure.py` | fig12_three_experiments (4-panel overview) |
| `make_each_experiment_figure.py` | exp1_accuracy, exp2_vision_dep, exp3_spatial_probe, exp23_combined |
| `make_scale_effect_figure.py` | fig7_scale_effect (500 vs 30k) |
| `make_scale_line_figure.py` | fig8_scale_line |
| `make_efficiency_map.py` | fig9_efficiency_map |
| `make_no_transfer_figure.py` | fig10_no_transfer |

## Result files
`compare_real_result.json` (9 variants), `compare_lowlr_diag.json` (lr 2e-5 stability),
`bench_inference_result.json`, `scaling_result.json`, `vision_dep_result.json`,
`spatial_probe_result.json`. Backups: `compare_real_result.{6var,7var,8var_3000}.json`.

## Open / in-progress
- `scaling_curve.py --variants maxinfo_scratch --sizes 500 30000` running to add
  maxinfo_scratch to fig7 (@30000 = 60k steps, ~hours). Regenerate fig7 when `scaling_result.json`
  has both `maxinfo_scratch@500` and `@30000`.

## Key numbers to sanity-check a rerun
frozen L1 0.0397 · honeybee L1 0.0477 / MSE 0.0213 / 64 tok / 260 ms · mlp_scratch_ln 0.0481 ·
self_attn 0.0534 · mlp_scratch collapse 0.2059 (lr 2e-4) → 0.071 (lr 2e-5) ·
vision_shift frozen 0.128 vs scratch_ln 0.002 · readout R² scratch_ln 0.42 > self_attn 0.28.
