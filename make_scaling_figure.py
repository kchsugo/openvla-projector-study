"""
maxinfo/make_scaling_figure.py

scaling_result.json -> 데이터 크기별 Action L1 추세선 그래프.
출력: maxinfo/figs/fig5_scaling_curve.png
실행: python maxinfo/make_scaling_figure.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
sc = json.load(open(os.path.join(HERE, "scaling_result.json")))

COL = {"baseline_mlp_frozen": "#2ca02c", "mlp_scratch": "#8c564b",
       "self_attn": "#d62728", "honeybee": "#ff7f0e",
       "cross_attn": "#9467bd", "maxinfo": "#1f77b4"}
SHORT = {"baseline_mlp_frozen": "MLP-frozen (사전학습, 고정)", "mlp_scratch": "MLP-scratch (랜덤)",
         "self_attn": "self-attn (랜덤)", "honeybee": "honeybee (랜덤)",
         "cross_attn": "cross-attn (랜덤)", "maxinfo": "maxinfo"}

variants, sizes = [], set()
for k in sc:
    nm, s = k.split("@"); sizes.add(int(s))
    if nm not in variants:
        variants.append(nm)
sizes = sorted(sizes)

plt.figure(figsize=(9, 5.8))
for nm in variants:
    xs, ys = [], []
    for s in sizes:
        k = f"{nm}@{s}"
        if k in sc:
            xs.append(s); ys.append(sc[k]["action_l1"])
    if not xs:
        continue
    style = "--" if nm == "baseline_mlp_frozen" else "-o"
    plt.plot(xs, ys, style, color=COL.get(nm, "#333"), lw=2,
             ms=6, label=SHORT.get(nm, nm))

plt.xscale("log")
plt.xlabel("학습 데이터 크기 (n_train, log scale)", fontsize=12)
plt.ylabel("Action L1 error  (낮을수록 좋음)", fontsize=12)
plt.title("Scaling Curve: 데이터가 늘수록 각 projector가 사전학습 MLP에 수렴하는가",
          fontsize=12.5)
plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=10, loc="best")
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_scaling_curve.png", dpi=200, bbox_inches="tight")
print("저장: figs/fig5_scaling_curve.png")
print(f"  변종 {len(variants)}종, 데이터 단계 {sizes}")
