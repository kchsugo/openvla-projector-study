"""
maxinfo/make_each_experiment_figure.py

교란 분리 3개 실험을 '각각' 개별 PNG로 (결과 분석/발표용, 한 장씩).
출력:
  figs/exp1_accuracy.png        Exp1 Action L1 (LN×mixing 2x2 통제)
  figs/exp2_vision_dep.png      Exp2 vision-dependency (image-swap shift)
  figs/exp3_spatial_probe.png   Exp3 spatial probe (action readout R²)
  figs/exp23_combined.png       결합 산점도 (Exp2 vs Exp3 분리도)
"""
import os, json, math
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

# ================= Exp1 — Action accuracy (LN × mixing 2x2) =================
fig, ax = plt.subplots(figsize=(11, 6.5))
order1 = ["mlp_scratch", "maxinfo_scratch", "mlp_scratch_ln", "self_attn", "honeybee"]
tags = {"mlp_scratch": "LN✗ mix✗", "maxinfo_scratch": "LN✗ mix✓",
        "mlp_scratch_ln": "LN✓ mix✗", "self_attn": "LN✓ mix✓", "honeybee": "LN✓ conv"}
vals = [exp1[k]["action_l1"] for k in order1]
bars = ax.bar(range(len(order1)), vals, color=[COL[k] for k in order1], width=0.62, zorder=3)
fr = exp1["baseline_mlp_frozen"]["action_l1"]
ax.axhline(fr, ls="--", color="#1a8a8a", lw=2, label=f"frozen baseline ({fr:.4f})")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(len(order1)))
ax.set_xticklabels([f"{LBL[k]}\n{tags[k]}" for k in order1], fontsize=11)
ax.set_ylabel("Action L1  (lower is better)", fontsize=12)
ax.set_title("Exp1 — Action accuracy: LayerNorm recovers collapse, spatial mixing does not",
             fontsize=13.5, fontweight="bold")
ax.set_ylim(0, max(vals)*1.2); ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
ax.annotate("", xy=(-0.4, 0.225), xytext=(1.4, 0.225),
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.5))
ax.text(0.5, 0.233, "No LayerNorm → COLLAPSE\n(spatial mixing can't save it)", ha="center",
        color="#c0392b", fontsize=11, fontweight="bold")
ax.annotate("", xy=(1.6, 0.085), xytext=(4.4, 0.085),
            arrowprops=dict(arrowstyle="-", color="#1e7a45", lw=1.5))
ax.text(3.0, 0.092, "With LayerNorm → recovered (mixing adds nothing)", ha="center",
        color="#1e7a45", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/exp1_accuracy.png", dpi=190, bbox_inches="tight"); plt.close()

# ================= Exp2 — vision-dependency =================
fig, ax = plt.subplots(figsize=(11, 6.5))
order2 = sorted([k for k in COL if k in exp2], key=lambda k: -exp2[k]["vision_shift"])
vals = [exp2[k]["vision_shift"] for k in order2]
bars = ax.bar(range(len(order2)), vals, color=[COL[k] for k in order2], width=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.0025, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(len(order2)))
ax.set_xticklabels([LBL[k] for k in order2], fontsize=11, rotation=12)
ax.set_ylabel("vision_shift  (prediction change when image is swapped)", fontsize=12)
ax.set_title("Exp2 — Visual grounding: frozen relies on vision the MOST;\n"
             "from-scratch variants barely react to the image (↑ = more vision-dependent)",
             fontsize=13.5, fontweight="bold")
ax.set_ylim(0, max(vals)*1.22); ax.grid(axis="y", alpha=0.3)
ax.text(0.5, 0.06, "← low: fits action priors, not vision", color="#b5651d",
        fontsize=10.5, style="italic")
plt.tight_layout(); plt.savefig(f"{FIG}/exp2_vision_dep.png", dpi=190, bbox_inches="tight"); plt.close()

# ================= Exp3 — spatial probe (readout R²) =================
fig, ax = plt.subplots(figsize=(11, 6.5))
order3 = [k for k in COL if k in exp3 and not math.isnan(exp3[k]["action_readout_r2"])]
order3 = sorted(order3, key=lambda k: -exp3[k]["action_readout_r2"])
vals = [exp3[k]["action_readout_r2"] for k in order3]
bars = ax.bar(range(len(order3)), vals, color=[COL[k] for k in order3], width=0.62, zorder=3)
ax.axhline(0, color="black", lw=0.9)
for b, v in zip(bars, vals):
    off = 0.2 if v >= 0 else -0.55
    ax.text(b.get_x()+b.get_width()/2, v+off, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(len(order3)))
ax.set_xticklabels([LBL[k] for k in order3], fontsize=11, rotation=12)
ax.set_ylabel("action readout R²  (higher = info more linearly accessible)", fontsize=12)
ax.set_title("Exp3 — Spatial/action info: mlp_scratch_ln (NO mixing) > self_attn\n"
             "LayerNorm drives accessibility  (mlp_scratch collapsed → NaN, omitted)",
             fontsize=13.5, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.text(0.98, 0.04, "weak proxy: jaco has no object-position labels (readout to GT action)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, style="italic", color="#888")
plt.tight_layout(); plt.savefig(f"{FIG}/exp3_spatial_probe.png", dpi=190, bbox_inches="tight"); plt.close()

# ================= 결합 산점도 Exp2 vs Exp3 =================
fig, ax = plt.subplots(figsize=(10, 7))
keys = [k for k in COL if k in exp2 and k in exp3 and not math.isnan(exp3[k]["action_readout_r2"])]
# frozen 과 maxinfo 는 좌표 동일(γ=0) → 하나로 병합 표기. 라벨 수동 오프셋으로 겹침 방지.
DISP = {"baseline_mlp_frozen": ("frozen = maxinfo (γ=0)", (12, -22), "left"),
        "maxinfo": (None, None, None),                       # frozen 과 동일점, 라벨 생략
        "honeybee": ("honeybee", (12, 8), "left"),
        "mlp_scratch_ln": ("mlp_scratch_ln", (12, 10), "left"),
        "self_attn": ("self_attn", (12, -16), "left")}
for k in keys:
    x, y = exp2[k]["vision_shift"], exp3[k]["action_readout_r2"]
    big = (k == "baseline_mlp_frozen")
    ax.scatter(x, y, s=440 if big else 300, color=COL[k], edgecolors="black",
               linewidths=1.6 if big else 1.1, zorder=3)
    lab, off, ha = DISP.get(k, (LBL[k], (10, 8), "left"))
    if lab:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                    fontsize=11, fontweight="bold", ha=ha)
ax.axhline(0, color="gray", lw=0.7, ls=":"); ax.axvline(0.06, color="gray", lw=0.7, ls=":")
ax.set_xlabel("Exp2: vision_shift  (→ more vision-dependent)", fontsize=12)
ax.set_ylabel("Exp3: action readout R²  (↑ more spatial/action info)", fontsize=12)
ax.set_title("Combined — two axes dissociate: 'carries action info' ≠ 'uses the image'",
             fontsize=13.5, fontweight="bold")
ax.set_xlim(-0.012, 0.15); ax.set_ylim(-7.6, 1.6)
ax.grid(alpha=0.3)
# 주석 박스는 점이 없는 빈 영역(중앙 상단)에 배치
ax.text(0.30, 0.97, "scratch_ln / self_attn:  high readout, LOW vision-use\n→ fit action priors, not the image",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#b5651d",
        bbox=dict(boxstyle="round", fc="#fff3e0", ec="#e67e22", alpha=0.9))
ax.text(0.97, 0.20, "frozen / maxinfo:\nmost vision-grounded",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#1e7a45",
        bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#2ca02c", alpha=0.9))
plt.tight_layout(); plt.savefig(f"{FIG}/exp23_combined.png", dpi=190, bbox_inches="tight"); plt.close()

print("저장 완료:")
for f in ["exp1_accuracy.png", "exp2_vision_dep.png", "exp3_spatial_probe.png", "exp23_combined.png"]:
    print("  figs/" + f)
