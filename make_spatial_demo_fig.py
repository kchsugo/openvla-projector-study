"""
maxinfo/make_spatial_demo_fig.py

"공간정보가 중요한 이미지" 데모 (재학습 없이 기존 5-seed 결과 활용).
scale_spatial_result.json 의 d6k_attn per-pair 연속 spatial_shift(5 seed) + frozen per-pair 를
읽고, 같은 선별 규칙으로 val 이미지를 복원해 다음을 한 그림에 그린다:
  (좌상) 64쌍 집계: frozen vs multiscale_attn 평균 pose shift (±seed std, p값)
  (좌하) per-seed 이득 분포(전부 양수면 효과 견고)
  (우)   seed-평균으로 multiscale_attn이 frozen보다 가장 크게 반응한 실제 이미지 쌍 예시

출력: maxinfo/figs/ms5_spatial_demo.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

from data import load_jaco_subset
from train_eval import CONFIG_PATH
from vision_dep_spatial import POSE

HERE = os.path.dirname(__file__); FIGS = os.path.join(HERE, "figs")
C_FZ = "#9aa0a6"; C_AT = "#d93025"


def select_spatial_idx(val_data, k=64, gripper_tol=0.3):
    """vision_dep_spatial.select_spatial_pairs 와 동일 순서의 (i,j) 인덱스."""
    acts = np.array([d[2] for d in val_data])
    cand = []
    for i in range(len(val_data)):
        for j in range(i + 1, len(val_data)):
            if abs(acts[i, 6] - acts[j, 6]) <= gripper_tol:
                cand.append((float(np.abs(acts[i, POSE] - acts[j, POSE]).mean()), i, j))
    cand.sort(reverse=True)
    return [(i, j) for _, i, j in cand[:k]]


def main():
    d = json.load(open(os.path.join(HERE, "scale_spatial_result.json")))["d6k_attn"]
    fr_pp = np.array(d["frozen"]["per_pair_cont"])                       # [64]
    seeds = sorted(d["runs"])
    at = np.array([d["runs"][k]["per_pair_cont"] for k in seeds])        # [S,64]
    at_avg = at.mean(0)                                                  # seed 평균 per-pair
    gains = np.array([(np.array(d["runs"][k]["per_pair_cont"]).mean() / fr_pp.mean() - 1) * 100 for k in seeds])
    p = ttest_1samp(gains, 0)[1]

    # val 이미지 복원 (train_n은 val에 무관 → 작게)
    print("loading val images…", flush=True)
    _, val_data = load_jaco_subset(10, 256, CONFIG_PATH, n_train_shards=1, n_test_shards=2)
    pairs = select_spatial_idx(val_data, k=len(fr_pp))

    # multiscale_attn이 frozen보다 가장 크게 반응한 쌍 top
    margin = at_avg - fr_pp
    order = np.argsort(margin)[::-1]
    top = order[:3]

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 1.1, 1.5])

    # (좌상) 집계 막대
    axA = fig.add_subplot(gs[0:1, 0:2])
    means = [fr_pp.mean(), at_avg.mean()]
    errs = [0, at.mean(1).std()]
    axA.bar(["frozen MLP", "multiscale_attn\n(5-seed avg)"], means, yerr=errs, capsize=5,
            color=[C_FZ, C_AT], alpha=0.9, edgecolor="black")
    for i, m in enumerate(means): axA.text(i, m, f" {m:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.set_ylabel("mean pose shift\n(continuous, 64 pairs)", fontsize=9)
    axA.set_title(f"Spatially-demanding images (k=64): multiscale_attn responds more\n"
                  f"+{gains.mean():.1f}% over frozen  (p={p:.3f}, 5 seeds)", fontsize=10, fontweight="bold")
    axA.grid(axis="y", alpha=0.25)

    # (좌하) per-seed 이득
    axB = fig.add_subplot(gs[1:3, 0:2])
    rng = np.random.default_rng(0)
    xs = 0.5 + (rng.random(len(gains)) - 0.5) * 0.25
    axB.scatter(xs, gains, s=80, color=C_AT, alpha=0.85, edgecolor="black", zorder=3)
    axB.plot([0.3, 0.7], [gains.mean()] * 2, color=C_AT, lw=2.5)
    axB.axhline(0, color="black", ls="--", lw=0.8)
    axB.set_xlim(0, 1); axB.set_xticks([0.5]); axB.set_xticklabels([f"per-seed gain (n={len(gains)})"])
    axB.set_ylabel("spatial-info gain vs frozen (%)", fontsize=9)
    axB.set_title(f"all {(gains>0).sum()}/{len(gains)} seeds positive — the effect is the aggregate, not any single run",
                  fontsize=9.5, fontweight="bold")
    axB.grid(axis="y", alpha=0.25)

    # (우) top 예시 이미지
    for r, idx in enumerate(top):
        i, j = pairs[idx]
        sub = gs[r, 2].subgridspec(1, 2, wspace=0.05)
        a0 = fig.add_subplot(sub[0]); a1 = fig.add_subplot(sub[1])
        a0.imshow(val_data[i][0]); a0.axis("off")
        a1.imshow(val_data[j][0]); a1.axis("off")
        if r == 0:
            a0.set_title("image A", fontsize=8); a1.set_title("image B (moved)", fontsize=8)
        a0.set_ylabel("")
        a1.text(1.02, 0.5, f"frozen {fr_pp[idx]:.3f}\nattn {at_avg[idx]:.3f}",
                transform=a1.transAxes, fontsize=7.5, va="center")

    fig.suptitle("Does the prediction track the spatial change? (multiscale_attn vs frozen MLP, real jaco images)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGS, "ms5_spatial_demo.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")
    print(f"frozen {fr_pp.mean():.4f} | attn(5-seed) {at_avg.mean():.4f} | gain +{gains.mean():.1f}% p={p:.3f} | pos {(gains>0).sum()}/{len(gains)}")


if __name__ == "__main__":
    main()
