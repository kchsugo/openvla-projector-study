"""
maxinfo/make_critique_figures.py  — 비판 대응(후속) 실험 시각화 3종.
  ms6_downstream_dissociation.png — hard/easy/all 에서 multiscale_attn Δaction-L1 vs frozen
  ms7_attn_ablation.png           — LLM 비전 attention 차단 시 frozen vs self_attn 악화 (인과)
  ms8_attn_map.png                — 층별 비전→비전 attention 질량 (관찰)
실행: python maxinfo/make_critique_figures.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

H = os.path.dirname(__file__); F = os.path.join(H, "figs")
C_OK = "#188038"; C_BAD = "#c5221f"; C_NEU = "#9aa0a6"; C_MS = "#1a73e8"


def fig_downstream():
    d = json.load(open(os.path.join(H, "spatial_action_result.json")))
    fr = d["frozen"]; runs = list(d["runs"].values())
    labels = [("hard\n(spatially-demanding)", "hard", C_OK),
              ("all\n(diluted by easy)", "all", C_NEU),
              ("easy\n(trivial states)", "easy", C_BAD)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(len(labels))
    for x, (lab, key, col) in zip(xs, labels):
        g = np.array([(r[key] - fr[key]) / fr[key] * 100 for r in runs])
        p = ttest_1samp(g, 0)[1]
        ax.bar(x, g.mean(), yerr=g.std(), capsize=5, color=col, alpha=0.9, edgecolor="black")
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."
        ax.text(x, g.mean() + np.sign(g.mean()) * (g.std() + 1.5),
                f"{g.mean():+.1f}%\n{star} (p={p:.3f})", ha="center",
                va="bottom" if g.mean() >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylim(-13, 33)
    ax.set_xticks(xs); ax.set_xticklabels([l[0] for l in labels], fontsize=9)
    ax.set_ylabel("multiscale_attn xyz action-L1\nvs frozen (%)   ↓ negative = better", fontsize=9.5)
    ax.set_title("Downstream action error: the gain is real but state-conditional\n"
                 "(helps spatially-demanding states, hurts trivial ones → flat on average)",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(F, "ms6_downstream_dissociation.png"), dpi=150); plt.close(fig)


def fig_ablation():
    d = json.load(open(os.path.join(H, "llm_attn_ablation_result.json")))
    names = [("frozen MLP", "baseline_mlp_frozen"), ("self_attn\n(projector pre-mixes)", "self_attn")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(names)); w = 0.36
    for i, (lab, key) in enumerate(names):
        v = d[key]
        ax.bar(i - w/2, v["normal_all"], w, color=C_MS, alpha=0.9, edgecolor="black",
               label="normal LLM" if i == 0 else None)
        ax.bar(i + w/2, v["block_all"], w, color=C_BAD, alpha=0.9, edgecolor="black",
               label="vision↔vision attn blocked" if i == 0 else None)
        ax.text(i + w/2, v["block_all"], f" +{v['degrade_all_%']:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_BAD)
        ax.text(i - w/2, v["normal_all"], f"{v['normal_all']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n[0] for n in names], fontsize=9.5)
    ax.set_ylabel("xyz action-L1 ↓", fontsize=10)
    ax.set_ylim(0, 0.105)
    ax.set_title("Causal test: blocking the LLM's vision↔vision attention\n"
                 "frozen collapses (+86%) — the LLM was doing the spatial mixing;\n"
                 "self_attn is immune (+2%) — projector mixing already did it",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", ncol=2); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(F, "ms7_attn_ablation.png"), dpi=150); plt.close(fig)


def fig_attnmap():
    d = json.load(open(os.path.join(H, "attn_map_result.json")))
    pl = d["per_layer"]; n = len(pl)
    vv = [x["vv_excl_self"] for x in pl]; bos = [x["vis_to_bos"] for x in pl]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    L = np.arange(n)
    ax.plot(L, vv, "-o", color=C_MS, lw=2, ms=4, label="vis→other-patches (spatial integration)")
    ax.plot(L, bos, "--", color=C_NEU, lw=1.5, label="vis→BOS (attention sink)")
    ax.fill_between(L, vv, alpha=0.15, color=C_MS)
    ax.set_xlabel("LLM layer", fontsize=10)
    ax.set_ylabel("avg attention mass from a vision token", fontsize=9.5)
    ax.set_title(f"Observational: vision tokens attend to OTHER patches\n"
                 f"(mean {d['summary']['vv_excl_self']['mean']:.2f}, peak {d['summary']['vv_excl_self']['max']:.2f}, "
                 f"strongest in early layers) → the LLM integrates space",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(F, "ms8_attn_map.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_downstream(); fig_ablation(); fig_attnmap()
    print("saved figs/ms6_downstream_dissociation.png ms7_attn_ablation.png ms8_attn_map.png")
