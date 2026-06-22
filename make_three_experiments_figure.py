"""
maxinfo/make_three_experiments_figure.py

교란 분리(disentangle) 체인 3개 실험을 한 장에 종합.
 Exp1  Action L1/MSE (compare_real_result.json)  — LN×mixing 2x2 통제
 Exp2  vision-dependency: image-swap shift (vision_dep_result.json)
 Exp3  spatial probe: action readout R² (spatial_probe_result.json)
 + 결합 산점도: Exp2(시각의존) vs Exp3(공간정보 readout) 분리도

출력: maxinfo/figs/fig12_three_experiments.png
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
exp1 = json.load(open(os.path.join(HERE, "compare_real_result.json")))
exp2 = json.load(open(os.path.join(HERE, "vision_dep_result.json")))
exp3 = json.load(open(os.path.join(HERE, "spatial_probe_result.json")))

COL = {"baseline_mlp_frozen": "#2ca02c", "mlp_scratch": "#c0392b",
       "maxinfo_scratch": "#e74c3c", "mlp_scratch_ln": "#27ae60",
       "self_attn": "#2980b9", "honeybee": "#e67e22", "maxinfo": "#7f7f7f"}
LBL = {"baseline_mlp_frozen": "frozen", "mlp_scratch": "mlp_scratch",
       "maxinfo_scratch": "maxinfo_scratch", "mlp_scratch_ln": "mlp_scratch_ln",
       "self_attn": "self_attn", "honeybee": "honeybee", "maxinfo": "maxinfo"}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Disentangle Experiments: Is it spatial mixing, or LayerNorm? (jaco_play)",
             fontsize=16, fontweight="bold")

# ---------- Exp1: Action L1 (LN x mixing 2x2 통제) ----------
ax = axes[0, 0]
order1 = ["mlp_scratch", "maxinfo_scratch", "mlp_scratch_ln", "self_attn", "honeybee"]
tags = {"mlp_scratch": "LN✗ mix✗", "maxinfo_scratch": "LN✗ mix✓",
        "mlp_scratch_ln": "LN✓ mix✗", "self_attn": "LN✓ mix✓", "honeybee": "LN✓ conv"}
vals = [exp1[k]["action_l1"] for k in order1]
bars = ax.bar(range(len(order1)), vals, color=[COL[k] for k in order1], zorder=3)
fr = exp1["baseline_mlp_frozen"]["action_l1"]
ax.axhline(fr, ls="--", color="#1a8a8a", lw=2, label=f"frozen ({fr:.4f})")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(order1)))
ax.set_xticklabels([f"{LBL[k]}\n{tags[k]}" for k in order1], fontsize=9)
ax.set_ylabel("Action L1 (lower better)", fontsize=11)
ax.set_title("Exp1 — Action accuracy: spatial mixing does NOT prevent collapse;\nLayerNorm does",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(vals)*1.18); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
ax.annotate("", xy=(-0.4, 0.225), xytext=(1.4, 0.225),
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.3))
ax.text(0.5, 0.232, "No LN → COLLAPSE\n(mixing can't save it)", ha="center",
        color="#c0392b", fontsize=9.5, fontweight="bold")

# ---------- Exp2: vision-dependency (image-swap shift) ----------
ax = axes[0, 1]
order2 = ["baseline_mlp_frozen", "honeybee", "mlp_scratch", "self_attn", "mlp_scratch_ln", "maxinfo"]
order2 = sorted([k for k in order2 if k in exp2], key=lambda k: -exp2[k]["vision_shift"])
vals = [exp2[k]["vision_shift"] for k in order2]
bars = ax.bar(range(len(order2)), vals, color=[COL[k] for k in order2], zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(order2)))
ax.set_xticklabels([LBL[k] for k in order2], fontsize=9, rotation=12)
ax.set_ylabel("vision_shift  (image-swap prediction change)", fontsize=11)
ax.set_title("Exp2 — Visual grounding: frozen uses vision the MOST;\nfrom-scratch variants barely react to the image",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(vals)*1.2); ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.95, "↑ higher = more vision-dependent", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, style="italic", color="#555")

# ---------- Exp3: spatial probe (action readout R²) ----------
ax = axes[1, 0]
order3 = ["mlp_scratch_ln", "self_attn", "honeybee", "baseline_mlp_frozen", "maxinfo"]  # mlp_scratch=NaN 제외
order3 = [k for k in order3 if k in exp3 and not math.isnan(exp3[k]["action_readout_r2"])]
order3 = sorted(order3, key=lambda k: -exp3[k]["action_readout_r2"])
vals = [exp3[k]["action_readout_r2"] for k in order3]
bars = ax.bar(range(len(order3)), vals, color=[COL[k] for k in order3], zorder=3)
ax.axhline(0, color="black", lw=0.8)
for b, v in zip(bars, vals):
    off = 0.15 if v >= 0 else -0.5
    ax.text(b.get_x()+b.get_width()/2, v+off, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(order3)))
ax.set_xticklabels([LBL[k] for k in order3], fontsize=9, rotation=12)
ax.set_ylabel("action readout R²  (higher = more linearly accessible)", fontsize=11)
ax.set_title("Exp3 — Spatial/action info readout: mlp_scratch_ln (NO mixing) > self_attn;\nLayerNorm drives accessibility (mlp_scratch = collapsed/NaN, omitted)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.05, "weak proxy: jaco has no object-position labels\n(readout to GT action)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, style="italic", color="#888")

# ---------- 결합 산점도: Exp2 vs Exp3 (분리도) ----------
ax = axes[1, 1]
keys = [k for k in COL if k in exp2 and k in exp3 and not math.isnan(exp3[k]["action_readout_r2"])]
for k in keys:
    x, y = exp2[k]["vision_shift"], exp3[k]["action_readout_r2"]
    ax.scatter(x, y, s=320, color=COL[k], edgecolors="black", linewidths=1.2, zorder=3)
    ax.annotate(LBL[k], (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")
ax.axhline(0, color="gray", lw=0.7, ls=":"); ax.axvline(0.06, color="gray", lw=0.7, ls=":")
ax.set_xlabel("Exp2: vision_shift  (→ more vision-dependent)", fontsize=11)
ax.set_ylabel("Exp3: action readout R²  (↑ more spatial/action info)", fontsize=11)
ax.set_title("Combined — two axes dissociate:\n'carries action info' ≠ 'uses the image'",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.text(0.02, 0.97, "scratch_ln/self_attn:\nhigh readout, LOW vision-use\n→ fit action priors, not vision",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#b5651d",
        bbox=dict(boxstyle="round", fc="#fff3e0", ec="#e67e22", alpha=0.8))
ax.text(0.98, 0.55, "frozen:\nmost vision-grounded",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#1e7a45",
        bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#2ca02c", alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{FIG}/fig12_three_experiments.png", dpi=170, bbox_inches="tight")
print("저장: figs/fig12_three_experiments.png")
print("Exp1 L1:", {k: round(exp1[k]["action_l1"], 4) for k in order1})
print("Exp2 vision_shift:", {k: round(exp2[k]["vision_shift"], 3) for k in order2})
print("Exp3 readout R²:", {k: round(exp3[k]["action_readout_r2"], 3) for k in order3})
