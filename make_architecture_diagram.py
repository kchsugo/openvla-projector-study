"""
maxinfo/make_architecture_diagram.py

논문 Method 섹션용 projector 아키텍처 비교 다이어그램 (모델 로드 불필요).
출력: maxinfo/figs/v3_architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)


def box(ax, x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=5)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=1.3, color=color))


fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

IN = "#d9e8fb"; MLP = "#cdeccd"; ATT = "#ffe0b3"; GATE = "#f5c6cb"; OUT = "#e0d4f7"

# ---- (a) 원본 MLP ----
ax = axes[0]; ax.set_title("(a) 원본 MLP projector\n= OpenVLA 기본", fontsize=11)
box(ax, 2.5, 8.3, 5, 1.1, "시각특징\n[256 × 2176]", IN)
box(ax, 2.5, 6.3, 5, 1.1, "fc1: 2176→8704 + GELU", MLP)
box(ax, 2.5, 4.6, 5, 1.1, "fc2: 8704→4096 + GELU", MLP)
box(ax, 2.5, 2.9, 5, 1.1, "fc3: 4096→4096", MLP)
box(ax, 2.5, 0.9, 5, 1.1, "LLM 입력\n[256 × 4096]", OUT)
for y1, y2 in [(8.3, 7.4), (6.3, 5.7), (4.6, 4.0), (2.9, 2.0)]:
    arrow(ax, 5, y1, 5, y2)
ax.text(5, 5.6, "토큰끼리 상호작용 없음\n(token-wise)", ha="center", fontsize=8.5,
        color="#b00", style="italic")

# ---- (b) honeybee / cross-attn (압축형) ----
ax = axes[1]; ax.set_title("(b) honeybee / cross-attn\n= 토큰 압축형", fontsize=11)
box(ax, 2.5, 8.3, 5, 1.1, "시각특징\n[256 × 2176]", IN)
box(ax, 2.5, 6.0, 5, 1.4, "압축\nConv / 64 query\n(LayerNorm 포함)", ATT)
box(ax, 2.5, 3.8, 5, 1.1, "Linear → 4096", MLP)
box(ax, 2.5, 1.6, 5, 1.1, "LLM 입력\n[64 × 4096]", OUT)
for y1, y2 in [(8.3, 7.4), (6.0, 4.9), (3.8, 2.7)]:
    arrow(ax, 5, y1, 5, y2)
ax.text(5, 0.7, "토큰 256→64 (정보 압축)", ha="center", fontsize=8.5, color="#b06000")

# ---- (c) self-attn ----
ax = axes[2]; ax.set_title("(c) self-attn\n= 토큰 관계 학습", fontsize=11)
box(ax, 2.5, 8.3, 5, 1.1, "시각특징\n[256 × 2176]", IN)
box(ax, 2.5, 6.0, 5, 1.4, "Self-Attention\n+ FFN\n(LayerNorm 포함)", ATT)
box(ax, 2.5, 3.8, 5, 1.1, "Linear → 4096", MLP)
box(ax, 2.5, 1.6, 5, 1.1, "LLM 입력\n[256 × 4096]", OUT)
for y1, y2 in [(8.3, 7.4), (6.0, 4.9), (3.8, 2.7)]:
    arrow(ax, 5, y1, 5, y2)
ax.text(5, 0.7, "토큰 간 attention (256 유지)", ha="center", fontsize=8.5, color="#b06000")

# ---- (d) maxinfo (제안) ----
ax = axes[3]; ax.set_title("(d) maxinfo (제안)\n= 보존 + 게이트 보강", fontsize=11)
box(ax, 0.6, 8.3, 8.8, 1.0, "시각특징 [256 × 2176]", IN)
# 두 갈래
box(ax, 0.4, 5.8, 4.0, 1.5, "Path A:\n원본 MLP\n(freeze, 계승)", MLP, fs=8)
box(ax, 5.6, 5.8, 4.0, 1.5, "Path B:\nself-attn ×2\n(enhancement)", ATT, fs=8)
box(ax, 6.0, 3.9, 3.2, 0.9, "× γ  (gate=0)", GATE, fs=8.5)
box(ax, 2.5, 1.9, 5, 1.1, "⊕  →  [256 × 4096]", OUT)
arrow(ax, 3.5, 8.3, 2.4, 7.3); arrow(ax, 6.5, 8.3, 7.6, 7.3)
arrow(ax, 2.4, 5.8, 4.0, 3.0)             # A -> sum
arrow(ax, 7.6, 5.8, 7.6, 4.8)             # B -> gate
arrow(ax, 7.6, 3.9, 6.0, 3.0)             # gate -> sum
ax.text(5, 0.9, "출력 = MLP(x) + γ·Enhance(x)\nγ=0 시작 → 원본과 동일(안전)",
        ha="center", fontsize=8.3, color="#7a0a2a")

fig.suptitle("OpenVLA Projector 구조 비교", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/v3_architecture.png", dpi=200, bbox_inches="tight")
print("저장: figs/v3_architecture.png")
