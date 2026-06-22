"""
maxinfo/make_scale_effect_figure.py

업로드된 'Training Step Effect' 그림과 동일 스타일로,
우리 scaling 실험의 데이터 크기 효과(@500 vs @30000)를 그룹 막대로 그린다.
출력: maxinfo/figs/fig7_scale_effect.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
sc = json.load(open(os.path.join(HERE, "scaling_result.json")))

VARIANTS = ["honeybee", "self_attn", "mlp_scratch"]   # scaling에 있는 랜덤 변종
SMALL, BIG = 500, 30000
frozen = sc[f"baseline_mlp_frozen@{SMALL}"]["action_l1"]   # 0.0397 (데이터 무관 고정)

small = [sc[f"{v}@{SMALL}"]["action_l1"] for v in VARIANTS]
big = [sc[f"{v}@{BIG}"]["action_l1"] for v in VARIANTS]

x = np.arange(len(VARIANTS)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 6.5))
b1 = ax.bar(x - w/2, small, w, label=f"{SMALL:,} samples", color="#E8A317")
b2 = ax.bar(x + w/2, big, w, label=f"{BIG:,} samples", color="#4DA6E8")
ax.axhline(frozen, ls="--", lw=2, color="#1a8a7a",
           label=f"Frozen MLP Baseline ({frozen:.4f})")

for bars in (b1, b2):
    for bi in bars:
        ax.text(bi.get_x()+bi.get_width()/2, bi.get_height(),
                f"{bi.get_height():.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(VARIANTS, fontsize=12)
ax.set_ylabel("Action L1 Error", fontsize=12)
ax.set_title("Fig: Data Scale Effect (500 vs 30,000 training samples)",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, max(max(small), max(big)) * 1.18)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_scale_effect.png", dpi=200, bbox_inches="tight")
print("저장: figs/fig7_scale_effect.png")
print(f"  frozen baseline={frozen:.4f}")
for v, s, b in zip(VARIANTS, small, big):
    print(f"  {v:<14} @500={s:.4f}  @30000={b:.4f}  변화={b-s:+.4f}")
