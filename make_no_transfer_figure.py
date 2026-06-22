"""
maxinfo/make_no_transfer_figure.py

'No Weight Transfer'(랜덤 초기화=사전학습 미계승) 변종들의 Action L1 가로 막대 비교.
각 변종은 안정적으로 학습된 setting의 값을 사용(붕괴한 scratch류는 lr 2e-5 진단값).
mlp_scratch_ln(LN 추가) 포함 — 최신화 버전.
출력: maxinfo/figs/fig10_no_transfer.png
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
main = json.load(open(os.path.join(HERE, "compare_real_result.json")))
diag = json.load(open(os.path.join(HERE, "compare_lowlr_diag.json")))
frozen = main["baseline_mlp_frozen"]["action_l1"]

# (라벨, L1, lr표기) — 랜덤 초기화 변종만. scratch류는 안정 setting(lr 2e-5).
rows = [
    ("maxinfo_scratch", diag["maxinfo_scratch"]["action_l1"], "lr 2e-5"),
    ("honeybee",        main["honeybee"]["action_l1"],        "lr 2e-4"),
    ("mlp_scratch_ln",  main["mlp_scratch_ln"]["action_l1"],  "lr 2e-4"),
    ("self_attn",       main["self_attn"]["action_l1"],       "lr 2e-4"),
    ("cross_attn",      main["cross_attn"]["action_l1"],      "lr 2e-4"),
    ("mlp_scratch",     diag["mlp_scratch"]["action_l1"],     "lr 2e-5"),
]
rows.sort(key=lambda r: r[1])                    # L1 오름차순(좋은 게 위)
labels = [f"{n}\n({lr})" for n, _, lr in rows]
vals = [v for _, v, _ in rows]
# 색: 위(좋음)→아래(나쁨) 그라데이션, mlp_scratch_ln만 회색 강조(대조군)
base_cols = ["#E8821E", "#1f77b4", "#7f7f7f", "#3a8ec4", "#9ecae1", "#9e9e9e"]
cols = []
for n, _, _ in rows:
    cols.append("#7f7f7f" if n == "mlp_scratch_ln" else None)
palette = ["#E8821E", "#1f77b4", "#4a90c2", "#7bb0d8", "#9e9e9e", "#bdbdbd"]
cols = [palette[i] for i in range(len(rows))]
# mlp_scratch_ln 강조색
for i, (n, _, _) in enumerate(rows):
    if n == "mlp_scratch_ln":
        cols[i] = "#555555"

y = range(len(rows))
fig, ax = plt.subplots(figsize=(11, 6.5))
bars = ax.barh(list(y), vals, color=cols, height=0.62, zorder=3)
ax.axvline(frozen, ls="--", lw=2.2, color="#1a8a8a",
           label=f"Frozen MLP Baseline ({frozen:.4f})")
for b, v in zip(bars, vals):
    ax.text(v + 0.0012, b.get_y() + b.get_height()/2, f"{v:.4f}",
            va="center", fontsize=11, fontweight="bold")
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=11)
ax.invert_yaxis()                                # 좋은 게 위로
ax.set_xlabel("Action L1 Error (Lower is Better)", fontsize=12)
ax.set_title("Fig: Projector Comparison (No Weight Transfer, 3000 Steps)",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, max(vals) * 1.25)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG}/fig10_no_transfer.png", dpi=200, bbox_inches="tight")
print("저장: figs/fig10_no_transfer.png")
for n, v, lr in rows:
    print(f"  {n:<16} {v:.4f} ({lr})")
