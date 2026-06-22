"""
maxinfo/make_ln_decomposition_figure.py

핵심 발견 시각화: scratch MLP의 회복은 '공간 mixing'이 아니라 'LayerNorm' 덕.
maxinfo_scratch(LN✗ 이지만 self-attn 공간 mixing 있음)를 추가해 LayerNorm×spatial-mixing
2x2 통제 비교로 만든다. 전부 동일 setting(lr 2e-4, 3000 step) → 교란 없음.
출력: maxinfo/figs/fig11_ln_decomposition.png
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
d = json.load(open(os.path.join(HERE, "compare_real_result.json")))
diag = json.load(open(os.path.join(HERE, "compare_lowlr_diag.json")))
frozen = d["baseline_mlp_frozen"]["action_l1"]

# 전부 lr 2e-4, 3000 step (동일 setting). maxinfo_scratch도 동일 setting=0.1944(붕괴).
rows = [
    ("mlp_scratch\n(LN ✗, mixing ✗)",       d["mlp_scratch"]["action_l1"],      "#c0392b"),
    ("maxinfo_scratch\n(LN ✗, spatial mix ✓)", d["maxinfo_scratch"]["action_l1"],  "#e74c3c"),
    ("mlp_scratch_ln\n(LN ✓, mixing ✗)",     d["mlp_scratch_ln"]["action_l1"],   "#27ae60"),
    ("self_attn\n(LN ✓, spatial mix ✓)",     d["self_attn"]["action_l1"],        "#2980b9"),
    ("honeybee\n(LN ✓, conv mix ✓)",         d["honeybee"]["action_l1"],         "#e67e22"),
]
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(13, 7.2))
x = range(len(rows))
bars = ax.bar(list(x), vals, color=cols, width=0.62, zorder=3)
ax.axhline(frozen, ls="--", lw=2, color="#1a8a8a", label=f"Frozen MLP baseline ({frozen:.4f})")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

# 그룹 묶음: 왼쪽 2개 = No LayerNorm (붕괴), 오른쪽 3개 = With LayerNorm (회복)
ax.annotate("", xy=(-0.32, 0.225), xytext=(1.32, 0.225),
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.4))
ax.text(0.5, 0.232, "No LayerNorm  →  COLLAPSE\n(spatial mixing can't save it)",
        ha="center", va="bottom", fontsize=11.5, color="#c0392b", fontweight="bold")
ax.annotate("", xy=(1.68, 0.085), xytext=(4.32, 0.085),
            arrowprops=dict(arrowstyle="-", color="#1e7a45", lw=1.4))
ax.text(3.0, 0.092, "With LayerNorm  →  RECOVERED  (mixing adds nothing)",
        ha="center", va="bottom", fontsize=11.5, color="#1e7a45", fontweight="bold")

# 화살표: +LayerNorm (붕괴 그룹 -> mlp_scratch_ln) 큰 회복
ax.add_patch(FancyArrowPatch((1, vals[1]*0.6), (2, vals[2]+0.025),
             arrowstyle="-|>", mutation_scale=24, lw=3, color="#27ae60",
             connectionstyle="arc3,rad=-0.25"))
ax.text(1.5, vals[1]*0.70, "+ LayerNorm\nrecovers collapse\n0.19 → 0.05",
        ha="center", fontsize=12, color="#1e7a45", fontweight="bold")

ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Action L1 Error  (lower is better)", fontsize=12)
ax.set_title("What recovers the scratch MLP? — LayerNorm, not spatial mixing\n"
             "(2×2 control: LayerNorm × spatial-mixing, all lr 2e-4, 3000 steps)",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, max(vals)*1.22)
ax.legend(fontsize=11, loc="center right")
ax.grid(axis="y", alpha=0.3)
# 각주: 매칭 lr 기준값 / lr 안정화 시 회복
fig.text(0.5, 0.005,
         "maxinfo_scratch shown at matched lr 2e-4 (same setting as others). "
         "With stabilized lr 2e-5 it recovers to 0.0474 ≈ mlp_scratch_ln — again, no LayerNorm → collapse.",
         ha="center", fontsize=9, color="#555", style="italic")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(f"{FIG}/fig11_ln_decomposition.png", dpi=200, bbox_inches="tight")
print("저장: figs/fig11_ln_decomposition.png")
for l, v, _ in rows: print(f"  {l.splitlines()[0]:<16} {v:.4f}")
