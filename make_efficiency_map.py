"""
maxinfo/make_efficiency_map.py

효율 지도: 정확도(Action L1) vs 추론 비용(LLM input tokens).
메시지: 64토큰(honeybee)은 정확도 손실이 작으면서 비용이 낮다 → on-device sweet spot.
출력: maxinfo/figs/fig9_efficiency_map.png
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
d = json.load(open(os.path.join(HERE, "compare_real_result.json")))
lat = json.load(open(os.path.join(HERE, "bench_inference_result.json")))

# 붕괴 변종(L1>0.15)은 효율 논의에서 제외(정상 동작하는 것만)
SHOW = {k: v for k, v in d.items() if v["action_l1"] < 0.15}
COL = {"baseline_mlp_frozen": "#2ca02c", "baseline_mlp_trained": "#98df8a",
       "honeybee": "#1f77b4", "self_attn": "#9467bd", "cross_attn": "#8c564b",
       "maxinfo": "#2ca02c", "mlp_scratch_ln": "#7f7f7f"}
LBL = {"baseline_mlp_frozen": "Frozen MLP", "baseline_mlp_trained": "MLP-trained",
       "honeybee": "honeybee", "self_attn": "self_attn", "cross_attn": "cross_attn",
       "maxinfo": "maxinfo", "mlp_scratch_ln": "mlp_scratch_ln"}

# 라벨 겹침 방지용 수동 오프셋 (frozen↔maxinfo 동일 위치, trained↔scratch_ln 인접)
OFF = {"baseline_mlp_frozen": (10, 14), "maxinfo": (10, -22),
       "baseline_mlp_trained": (-10, 16), "mlp_scratch_ln": (12, 16),
       "self_attn": (12, -20), "honeybee": (14, 10), "cross_attn": (12, 12)}

fig, ax = plt.subplots(figsize=(12, 7))
for k, v in SHOW.items():
    x, y = v["action_l1"], v["tokens"]
    big = (k == "honeybee")
    ax.scatter(x, y, s=620 if big else 320, color=COL.get(k, "#333"),
               edgecolors="black", linewidths=1.6 if big else 1.0, zorder=3, alpha=0.9)
    ox, oy = OFF.get(k, (12, 12))
    ha = "right" if ox < 0 else "left"
    ax.annotate(LBL.get(k, k), (x, y), textcoords="offset points",
                xytext=(ox, oy), fontsize=11, ha=ha,
                fontweight="bold" if big else "normal")

# 추론 latency 주석 (측정된 것)
for k in ("baseline_mlp_frozen", "honeybee", "cross_attn"):
    if k in SHOW and k in lat:
        ax.annotate(f"{lat[k]['latency_ms']:.0f}ms", (SHOW[k]['action_l1'], SHOW[k]['tokens']),
                    textcoords="offset points", xytext=(12, -20), fontsize=8.5, color="#555")

# 고효율 영역(좌하단: 정확하고 비용 낮음)
ax.add_patch(FancyBboxPatch((0.035, 5), 0.025, 95, boxstyle="round,pad=0.002",
             fill=False, ec="red", lw=1.6, ls="-"))
ax.text(0.0475, 100, "High-Efficiency Region\n(accurate & low-cost)",
        color="red", fontsize=10.5, ha="center", va="bottom", fontweight="bold")

ax.set_xlabel("Action L1 Error  (← more accurate)", fontsize=12)
ax.set_ylabel("LLM Input Tokens  (↓ lower inference cost)", fontsize=12)
ax.set_title("Fig: Efficiency Map — Accuracy vs Inference Cost", fontsize=14, fontweight="bold")
ax.set_xlim(0.033, 0.075); ax.set_ylim(0, 290)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG}/fig9_efficiency_map.png", dpi=200, bbox_inches="tight")
print("저장: figs/fig9_efficiency_map.png")
fr = SHOW["baseline_mlp_frozen"]; hb = SHOW["honeybee"]
print(f"  honeybee: 64tok(-75%), L1 {hb['action_l1']:.4f} vs frozen {fr['action_l1']:.4f} "
      f"(차이 {hb['action_l1']-fr['action_l1']:+.4f}), latency {lat['honeybee']['latency_ms']:.0f}ms vs {lat['baseline_mlp_frozen']['latency_ms']:.0f}ms")
