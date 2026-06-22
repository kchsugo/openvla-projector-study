"""
maxinfo/make_figures.py

compare_real_result.json + train_full.log 을 읽어 논문용 figure 생성.
출력: maxinfo/figs/*.png
"""
import os, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

ORDER = ["baseline_mlp_frozen", "baseline_mlp_trained", "mlp_scratch", "mlp_scratch_ln",
         "honeybee", "self_attn", "cross_attn", "maxinfo", "maxinfo_scratch"]
SHORT = {"baseline_mlp_frozen": "MLP-frozen", "baseline_mlp_trained": "MLP-trained",
         "mlp_scratch": "MLP-scratch", "mlp_scratch_ln": "MLP-scratch+LN",
         "honeybee": "honeybee", "self_attn": "self-attn",
         "cross_attn": "cross-attn", "maxinfo": "maxinfo", "maxinfo_scratch": "maxinfo-scratch"}
COL = {"baseline_mlp_frozen": "#2ca02c", "baseline_mlp_trained": "#98df8a",
       "mlp_scratch": "#8c564b", "mlp_scratch_ln": "#7f7f7f", "honeybee": "#ff7f0e",
       "self_attn": "#d62728", "cross_attn": "#9467bd", "maxinfo": "#1f77b4",
       "maxinfo_scratch": "#17becf"}

res = json.load(open(os.path.join(HERE, "compare_real_result.json")))
names = [n for n in ORDER if n in res]
labels = [SHORT[n] for n in names]
colors = [COL[n] for n in names]
l1 = [res[n]["action_l1"] for n in names]
acc = [res[n]["token_acc"] for n in names]
mse = [res[n]["action_mse"] for n in names]
params = [res[n]["trainable_params"] / 1e6 for n in names]

# ---- Fig 1: action L1 (낮을수록 좋음) ----
plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, l1, color=colors)
plt.axhline(l1[names.index("baseline_mlp_frozen")], ls="--", c="gray", lw=1,
            label="frozen baseline")
for b, v in zip(bars, l1):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
plt.ylabel("Action L1 error  (lower is better)")
plt.title("Fig 1. Action L1 by projector variant (jaco_play, 1500 samples, 3000 steps)")
plt.legend(); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(f"{FIG}/fig1_action_l1.png", dpi=150); plt.close()

# ---- Fig 2: token acc vs action L1 (역설 강조) ----
fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(names))
ax1.bar(x - 0.2, acc, 0.4, color="#4c72b0", label="token acc (↑)")
ax1.set_ylabel("Token accuracy (↑)", color="#4c72b0")
ax1.set_ylim(0.6, 0.82)
ax2 = ax1.twinx()
ax2.bar(x + 0.2, l1, 0.4, color="#dd8452", label="action L1 (↓)")
ax2.set_ylabel("Action L1 (↓)", color="#dd8452")
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15)
plt.title("Fig 2. Token accuracy vs Action L1 (higher acc ≠ lower L1)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_acc_vs_l1.png", dpi=150); plt.close()

# ---- Fig 3: params vs L1 (파라미터 많다고 좋은 게 아님) ----
plt.figure(figsize=(7, 5))
for n in names:
    plt.scatter(res[n]["trainable_params"] / 1e6, res[n]["action_l1"],
                s=120, color=COL[n], label=SHORT[n], zorder=3)
    plt.annotate(SHORT[n], (res[n]["trainable_params"] / 1e6, res[n]["action_l1"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.axhline(l1[names.index("baseline_mlp_frozen")], ls="--", c="gray", lw=1)
plt.xlabel("Trainable params (M)"); plt.ylabel("Action L1 (↓)")
plt.title("Fig 3. More parameters do not help (maxinfo 114M → γ=0 → = frozen)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_params_vs_l1.png", dpi=150); plt.close()

# ---- Fig 4: training loss curves (로그 파싱) ----
log = os.path.join(HERE, "train_3000.log")
curves, cur = {}, None
if os.path.exists(log):
    for line in open(log, errors="ignore"):
        m = re.search(r"=====\s*(\S+)\s*=====", line)
        if m:
            cur = m.group(1); curves[cur] = ([], [])
        m = re.search(r"step (\d+)/\d+ loss=([\d.]+)", line)
        if m and cur:
            curves[cur][0].append(int(m.group(1)))
            curves[cur][1].append(float(m.group(2)))
    plt.figure(figsize=(8, 5))
    for n in names:
        if n in curves and curves[n][0]:
            plt.plot(curves[n][0], curves[n][1], marker="o", ms=3,
                     color=COL[n], label=SHORT[n])
    plt.xlabel("training step"); plt.ylabel("loss")
    plt.title("Fig 4. Training loss curves (3000 steps)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{FIG}/fig4_loss_curves.png", dpi=150); plt.close()

print("saved figures to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f)
