"""
maxinfo/make_korean_figures.py — 논문용 그림 3종 한글판(영어 원본 보존, _ko 접미사).
  ms6_ko.png  다운스트림 행동오차 dissociation
  ms7_ko.png  LLM 비전 attention 차단 인과검증
  fig_scratch_ko.png  무엇이 scratch MLP를 회복시키나 (LayerNorm vs 공간혼합)
폰트: Noto Sans CJK JP (한글 포함). 실행: python maxinfo/make_korean_figures.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import ttest_1samp

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

H = os.path.dirname(__file__); F = os.path.join(H, "figs")
C_OK = "#188038"; C_BAD = "#c5221f"; C_NEU = "#9aa0a6"; C_MS = "#1a73e8"


def ms6_ko():
    d = json.load(open(os.path.join(H, "spatial_action_result.json")))
    fr = d["frozen"]; runs = list(d["runs"].values())
    labels = [("hard\n(공간 중요)", "hard", C_OK),
              ("all\n(easy에 희석)", "all", C_NEU),
              ("easy\n(자명한 상태)", "easy", C_BAD)]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_ylim(-13, 33)
    for x, (lab, key, col) in enumerate(labels):
        g = np.array([(r[key] - fr[key]) / fr[key] * 100 for r in runs])
        p = ttest_1samp(g, 0)[1]
        ax.bar(x, g.mean(), yerr=g.std(), capsize=5, color=col, alpha=0.9, edgecolor="black")
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."
        ax.text(x, g.mean() + np.sign(g.mean()) * (g.std() + 1.5), f"{g.mean():+.1f}%\n{star} (p={p:.3f})",
                ha="center", va="bottom" if g.mean() >= 0 else "top", fontsize=9.5, fontweight="bold")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels([l[0] for l in labels], fontsize=9.5)
    ax.set_ylabel("multiscale_attn xyz 행동 L1\nfrozen 대비 (%)   ↓ 음수 = 개선", fontsize=10)
    ax.set_title("다운스트림 행동 오차: 이득은 실재하나 상태 조건부\n"
                 "(공간이 중요한 상태는 개선, 자명한 상태는 악화 → 평균은 상쇄)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(F, "ms6_ko.png"), dpi=150); plt.close(fig)


def ms7_ko():
    d = json.load(open(os.path.join(H, "llm_attn_ablation_result.json")))
    names = [("동결 MLP", "baseline_mlp_frozen"), ("self_attn\n(projector가 미리 섞음)", "self_attn")]
    fig, ax = plt.subplots(figsize=(7.4, 4.8)); w = 0.36
    ax.set_ylim(0, 0.105)
    for i, (lab, key) in enumerate(names):
        v = d[key]
        ax.bar(i - w/2, v["normal_all"], w, color=C_MS, alpha=0.9, edgecolor="black",
               label="정상 LLM" if i == 0 else None)
        ax.bar(i + w/2, v["block_all"], w, color=C_BAD, alpha=0.9, edgecolor="black",
               label="비전↔비전 어텐션 차단" if i == 0 else None)
        ax.text(i + w/2, v["block_all"], f" +{v['degrade_all_%']:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=C_BAD)
        ax.text(i - w/2, v["normal_all"], f"{v['normal_all']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(2)); ax.set_xticklabels([n[0] for n in names], fontsize=9.5)
    ax.set_ylabel("xyz 행동 L1 ↓", fontsize=10)
    ax.set_title("인과 검증: LLM의 비전↔비전 어텐션 차단\n"
                 "동결 MLP는 86% 붕괴 — LLM이 공간 혼합을 수행했음;\n"
                 "self_attn은 +2%로 영향 없음 — projector가 이미 섞었음",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper center", ncol=2); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(F, "ms7_ko.png"), dpi=150); plt.close(fig)


def scratch_ko():
    d = json.load(open(os.path.join(H, "compare_real_result.json")))
    frozen = d["baseline_mlp_frozen"]["action_l1"]
    rows = [("mlp_scratch\n(LN X, 혼합 X)", d["mlp_scratch"]["action_l1"], C_BAD),
            ("mlp_scratch_ln\n(LN O, 혼합 X)", d["mlp_scratch_ln"]["action_l1"], C_OK),
            ("self_attn\n(LN O, 공간혼합 O)", d["self_attn"]["action_l1"], C_MS),
            ("honeybee\n(LN O, conv혼합 O)", d["honeybee"]["action_l1"], "#e67e22")]
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    bars = ax.bar(range(len(rows)), vals, color=cols, width=0.6, zorder=3)
    ax.axhline(frozen, ls="--", lw=2, color="#1a8a8a", label=f"동결 MLP 기준선 ({frozen:.4f})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.4f}", ha="center", fontsize=13, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0, vals[0]*0.6), (1, vals[1]+0.03), arrowstyle="-|>",
                 mutation_scale=22, lw=3, color=C_OK, connectionstyle="arc3,rad=-0.3"))
    ax.text(0.5, vals[0]*0.66, "+ LayerNorm\n붕괴 회복\n0.2059 → 0.0481",
            ha="center", fontsize=12, color="#1e7a45", fontweight="bold")
    ax.add_patch(FancyArrowPatch((1, vals[1]+0.012), (2, vals[2]+0.012), arrowstyle="-|>",
                 mutation_scale=18, lw=2, color=C_NEU, connectionstyle="arc3,rad=-0.5"))
    ax.text(1.5, vals[1]+0.045, "+ 공간 혼합\n이득 없음 (오히려 악화)\n0.0481 → 0.0534",
            ha="center", fontsize=11, color="#555", fontweight="bold")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("행동 L1 오차 (낮을수록 좋음)", fontsize=12)
    ax.set_title("무엇이 scratch MLP를 회복시키나? — 공간 혼합이 아니라 LayerNorm",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.2); ax.legend(fontsize=11, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_scratch_ko.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    ms6_ko(); ms7_ko(); scratch_ko()
    print("저장: figs/ms6_ko.png figs/ms7_ko.png figs/fig_scratch_ko.png")
