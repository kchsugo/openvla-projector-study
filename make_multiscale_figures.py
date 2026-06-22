"""
maxinfo/make_multiscale_figures.py

§6(정보 전달 / 멀티스케일) 전용 그림 세트. 액션 성능은 다루지 않는다.
  ms1_architecture.png  — 데이터 흐름: 원본(끝-2 한 층) vs multiscale vs multiscale_attn
  ms2_gain.png          — frozen 대비 spatial_shift 상대이득(±std, 유의성)
  ms3_metric.png        — argmax(잡음) vs 연속(깨끗) per-seed 분포
  ms4_deadlock.png      — maxinfo 이중 zero-init 데드락(γ·loss)

실행: python maxinfo/make_multiscale_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

C_FROZEN = "#9aa0a6"; C_MS = "#1a73e8"; C_ATTN = "#d93025"; C_OK = "#188038"; C_BAD = "#c5221f"


def _gains(per_pairs, base):
    base_m = np.mean(base)
    return np.array([np.mean(p) / base_m * 100 - 100 for p in per_pairs])


def load():
    d6 = json.load(open(os.path.join(HERE, "scale_spatial_result.json")))
    v1 = json.load(open(os.path.join(HERE, "vision_dep_spatial_result.json")))
    out = {}
    # multiscale 1.5k (argmax only, 10 seeds)
    base1 = np.array(v1["baseline_mlp_frozen"]["per_pair"])
    ks = sorted([k for k in v1 if k.startswith("multiscale_seed")], key=lambda x: int(x.split("seed")[1]))
    out["ms_1k5_argmax"] = _gains([v1[k]["per_pair"] for k in ks], base1)
    # multiscale 6k (argmax + cont)
    c = d6["d6k_s3000"]; fr = c["frozen"]; runs = c["runs"]; rk = sorted(runs)
    out["ms_6k_argmax"] = _gains([runs[k]["per_pair"] for k in rk], fr["per_pair"])
    out["ms_6k_cont"] = _gains([runs[k]["per_pair_cont"] for k in rk], fr["per_pair_cont"])
    # multiscale_attn 1.5k (argmax + cont)
    if "d1k5" in d6:
        c = d6["d1k5"]; fr = c["frozen"]; runs = c["runs"]; rk = sorted(runs)
        out["attn_1k5_argmax"] = _gains([runs[k]["per_pair"] for k in rk], fr["per_pair"])
        out["attn_1k5_cont"] = _gains([runs[k]["per_pair_cont"] for k in rk], fr["per_pair_cont"])
    # multiscale_attn 6k (if present)
    if "d6k_attn" in d6 and d6["d6k_attn"].get("runs"):
        c = d6["d6k_attn"]; fr = c["frozen"]; runs = c["runs"]; rk = sorted(runs)
        out["attn_6k_cont"] = _gains([runs[k]["per_pair_cont"] for k in rk], fr["per_pair_cont"])
    return out


def _ttest_p(a):
    from scipy.stats import ttest_1samp
    return ttest_1samp(a, 0.0)[1] if len(a) > 1 else float("nan")


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


# ---------- Fig MS2: gain bars ----------
def fig_gain(g):
    rows = [("multiscale\n1.5k · argmax", "ms_1k5_argmax", C_MS),
            ("multiscale\n6k · argmax", "ms_6k_argmax", C_MS),
            ("multiscale\n6k · continuous", "ms_6k_cont", C_MS),
            ("multiscale_attn\n1.5k · continuous", "attn_1k5_cont", C_ATTN)]
    if "attn_6k_cont" in g:
        rows.append(("multiscale_attn\n6k · continuous", "attn_6k_cont", C_ATTN))
    labels = [r[0] for r in rows]; arrs = [g[r[1]] for r in rows]; cols = [r[2] for r in rows]
    means = [a.mean() for a in arrs]; stds = [a.std() for a in arrs]; ps = [_ttest_p(a) for a in arrs]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=cols, alpha=0.88, edgecolor="black", linewidth=0.6)
    for xi, (m, s, p, a) in enumerate(zip(means, stds, ps, arrs)):
        ax.text(xi, m + s + 1.5, f"{m:+.1f}%\n{_stars(p)} (n={len(a)})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("spatial information transferred\nvs frozen MLP  (Δ spatial_shift, %)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("Multi-scale projectors transfer significantly more spatial information to the LLM", fontsize=11, fontweight="bold")
    ax.set_ylim(top=max(m + s for m, s in zip(means, stds)) + 12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "ms2_gain.png"), dpi=150); plt.close(fig)


# ---------- Fig MS3: metric matters ----------
def fig_metric(g):
    a = g["ms_6k_argmax"]; c = g["ms_6k_cont"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    rng = np.random.default_rng(0)
    lo = min(a.min(), c.min()); hi = max(a.max(), c.max())
    ax.set_ylim(lo - 12, hi + 16)
    for xi, (arr, col) in enumerate([(a, C_BAD), (c, C_OK)]):
        xs = xi + (rng.random(len(arr)) - 0.5) * 0.18
        ax.scatter(xs, arr, s=60, color=col, alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
        ax.plot([xi - 0.22, xi + 0.22], [arr.mean()] * 2, color=col, lw=2.5, zorder=4)
        ax.text(xi, hi + 9, f"μ={arr.mean():+.1f}%\nσ={arr.std():.1f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=col)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks([0, 1]); ax.set_xlim(-0.5, 1.5)
    ax.set_xticklabels(["argmax\n(256-bin, quantized)", "continuous\n(softmax expectation)"], fontsize=9)
    ax.set_ylabel("per-seed gain vs frozen (%)", fontsize=10)
    ax.set_title("Same runs, two metrics:\nargmax buries the signal in quantization noise",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "ms3_metric.png"), dpi=150); plt.close(fig)


# ---------- Fig MS4: deadlock ----------
def fig_deadlock():
    mf = json.load(open(os.path.join(HERE, "compare_maxinfo_fixed.json")))
    names = ["maxinfo\n(orig, deadlocked)", "maxinfo_fixed\n(deadlock removed)"]
    gam = [abs(mf["maxinfo"]["gamma"]), abs(mf["maxinfo_fixed"]["gamma"])]
    loss = [mf["maxinfo"]["final_loss"], mf["maxinfo_fixed"]["final_loss"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.9))
    a1.bar(names, gam, color=[C_BAD, C_OK], alpha=0.88, edgecolor="black")
    a1.set_title("gate |γ| (final)", fontsize=10, fontweight="bold"); a1.set_ylabel("|γ|")
    for i, v in enumerate(gam): a1.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    a2.bar(names, loss, color=[C_BAD, C_OK], alpha=0.88, edgecolor="black")
    a2.set_title("final train loss", fontsize=10, fontweight="bold"); a2.set_ylabel("loss")
    for i, v in enumerate(loss): a2.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle("Original maxinfo never trains (γ≡0, loss flat): a gradient deadlock, not 'no-regret'", fontsize=10.5, fontweight="bold")
    for ax in (a1, a2): ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "ms4_deadlock.png"), dpi=150); plt.close(fig)


# ---------- Fig MS1: architecture ----------
def fig_arch():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    titles = ["(a) OpenVLA original", "(b) MultiScale (more input)", "(c) MultiScale-Attn (fuse scales)"]

    def box(ax, x, y, w, h, text, color, fc=None, alpha=1.0, fs=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.1, edgecolor="black", facecolor=fc or color, alpha=alpha)
        ax.add_patch(p); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(ax, x0, y0, x1, y1, color="black", style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, lw=1.3, color=color))

    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        # ViT layer stack
        ys = [0.80, 0.66, 0.52, 0.38]
        for i, y in enumerate(ys):
            picked = (i == 0)
            box(ax, 0.06, y, 0.20, 0.10,
                ("ViT layer  (penult)" if picked else "ViT layer"),
                C_MS if picked else "#e8eaed",
                fc=("#cfe2ff" if picked else "#e8eaed"),
                alpha=1.0, fs=7)

    # (a) only penult used; rest discarded
    ax = axes[0]
    for y in [0.66, 0.52, 0.38]:
        ax.text(0.295, y + 0.05, "✗ discarded", color=C_BAD, fontsize=7.5, va="center")
    box(ax, 0.46, 0.74, 0.18, 0.12, "MLP", C_FROZEN, fc="#dadce0", fs=9)
    box(ax, 0.78, 0.74, 0.16, 0.12, "LLM", "#fde293", fc="#fde293", fs=9)
    arrow(ax, 0.26, 0.85, 0.46, 0.80); arrow(ax, 0.64, 0.80, 0.78, 0.80)
    ax.text(0.5, 0.2, "only 1 layer reaches the LLM", fontsize=8.5, ha="center", color=C_BAD)

    # (b) multiscale: penult + intermediate -> base + gamma*Enhance
    ax = axes[1]
    box(ax, 0.46, 0.78, 0.18, 0.10, "frozen MLP", C_FROZEN, fc="#dadce0", fs=8)
    box(ax, 0.46, 0.50, 0.18, 0.10, "Enhance\n(token-wise)", C_MS, fc="#cfe2ff", fs=7.5)
    box(ax, 0.70, 0.64, 0.10, 0.10, "+ γ·", "#ffffff", fc="#ffffff", fs=10)
    box(ax, 0.84, 0.64, 0.13, 0.10, "LLM", "#fde293", fc="#fde293", fs=9)
    arrow(ax, 0.26, 0.85, 0.46, 0.83)                 # penult -> MLP
    arrow(ax, 0.26, 0.55, 0.46, 0.55)                 # mid -> Enhance
    ax.text(0.30, 0.49, "+ intermediate\n(was discarded)", color=C_OK, fontsize=7, va="center")
    arrow(ax, 0.64, 0.83, 0.70, 0.71); arrow(ax, 0.64, 0.55, 0.70, 0.68)
    arrow(ax, 0.80, 0.69, 0.84, 0.69)

    # (c) multiscale_attn: scales fused per patch by attention
    ax = axes[2]
    box(ax, 0.46, 0.78, 0.18, 0.10, "frozen MLP", C_FROZEN, fc="#dadce0", fs=8)
    box(ax, 0.44, 0.46, 0.22, 0.12, "cross-scale\nattention\n(per patch)", C_ATTN, fc="#fad2cf", fs=7.5)
    box(ax, 0.72, 0.62, 0.10, 0.10, "+ γ·", "#ffffff", fc="#ffffff", fs=10)
    box(ax, 0.86, 0.62, 0.12, 0.10, "LLM", "#fde293", fc="#fde293", fs=9)
    arrow(ax, 0.26, 0.85, 0.46, 0.83)
    for y in [0.80, 0.66, 0.52, 0.38]:
        arrow(ax, 0.26, y + 0.05, 0.44, 0.54, color="#bbbbbb")
    ax.text(0.29, 0.40, "all scales\n→ attention", color=C_ATTN, fontsize=7, va="center")
    arrow(ax, 0.64, 0.83, 0.72, 0.69); arrow(ax, 0.66, 0.52, 0.72, 0.66)
    arrow(ax, 0.82, 0.67, 0.86, 0.67)

    fig.suptitle("Lever = which features reach the LLM (and fusing them across scale), not mixing patches the LLM already mixes",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGS, "ms1_architecture.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    g = load()
    fig_arch()
    fig_gain(g)
    fig_metric(g)
    fig_deadlock()
    print("saved: figs/ms1_architecture.png ms2_gain.png ms3_metric.png ms4_deadlock.png")
    for k, v in g.items():
        print(f"  {k}: mean={v.mean():+.1f}% std={v.std():.1f} n={len(v)}")
