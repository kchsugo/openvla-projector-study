"""
maxinfo/make_scale_line_figure.py

세 변종(honeybee, self_attn, mlp_scratch)의 데이터 크기별 L1 변화를 선그래프로.
전체 5단계(500→2000→5000→10000→30000) 추이 + frozen 기준선(점선).
출력: maxinfo/figs/fig8_scale_line.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
sc = json.load(open(os.path.join(HERE, "scaling_result.json")))

SIZES = [500, 2000, 5000, 10000, 30000]
VARIANTS = ["self_attn", "honeybee", "mlp_scratch"]
STYLE = {
    "self_attn":   ("#d62728", "o", "self_attn (spatial mixing)"),
    "honeybee":    ("#ff7f0e", "s", "honeybee (conv compress)"),
    "mlp_scratch": ("#8c564b", "^", "mlp_scratch (no mixing, no LN)"),
}
frozen = sc[f"baseline_mlp_frozen@{SIZES[0]}"]["action_l1"]

fig, ax = plt.subplots(figsize=(11, 6.5))
for v in VARIANTS:
    ys = [sc[f"{v}@{s}"]["action_l1"] for s in SIZES]
    c, mk, lbl = STYLE[v]
    ax.plot(SIZES, ys, marker=mk, ms=9, lw=2.5, color=c, label=lbl)
    for s, y in zip(SIZES, ys):
        ax.annotate(f"{y:.3f}", (s, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=c, fontweight="bold")

ax.axhline(frozen, ls="--", lw=2, color="#1a8a7a",
           label=f"Frozen MLP baseline ({frozen:.4f})")
ax.set_xscale("log")
ax.set_xticks(SIZES); ax.set_xticklabels([f"{s:,}" for s in SIZES], fontsize=11)
ax.set_xlabel("Training data size (n_train, log scale)", fontsize=12)
ax.set_ylabel("Action L1 Error  (lower is better)", fontsize=12)
ax.set_title("Fig: Data Scaling of Projector Variants (500 → 30,000)",
             fontsize=14, fontweight="bold")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIG}/fig8_scale_line.png", dpi=200, bbox_inches="tight")
print("저장: figs/fig8_scale_line.png")
for v in VARIANTS:
    ys = [sc[f"{v}@{s}"]['action_l1'] for s in SIZES]
    print(f"  {v:<14} " + " ".join(f"{s}:{y:.3f}" for s, y in zip(SIZES, ys)))
