"""
maxinfo/spatial_demo.py

"공간정보가 중요한 이미지"에서의 정성 데모.
같은 지시문(text A)을 두고 이미지만 A→B(=물체/그리퍼 위치가 크게 다른 다른 에피소드)로 바꿨을 때,
frozen MLP vs multiscale_attn 의 예측 액션이 그 공간변화를 얼마나 따라 움직이는지(pose shift)를
실제 이미지와 함께 그린다. 참조선 = GT pose shift(이미지가 실제로 요구하는 변화량).

실행: python maxinfo/spatial_demo.py --train_n 1500 --steps 800 --k 6
출력: maxinfo/figs/ms5_spatial_demo.png
"""
import os, sys, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from data import load_jaco_subset
from projectors_zoo import build_projector
from train_eval import MODEL_ID, CONFIG_PATH, load_norm, build_examples, train, decode_tokens
from vision_dep_spatial import predict_continuous, POSE

FIGS = os.path.join(os.path.dirname(__file__), "figs")


def select_spatial_idx(val_data, k=6, gripper_tol=0.3):
    acts = np.array([d[2] for d in val_data])
    cand = []
    for i in range(len(val_data)):
        for j in range(i + 1, len(val_data)):
            if abs(acts[i, 6] - acts[j, 6]) <= gripper_tol:
                cand.append((np.abs(acts[i, POSE] - acts[j, POSE]).mean(), i, j))
    cand.sort(reverse=True)
    return [(i, j) for _, i, j in cand[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=["projector"])
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, quantization_config=qc, device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    model.gradient_checkpointing_enable()
    orig_base = model.projector
    vocab_size = model.vocab_size
    norm = load_norm()

    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH, n_train_shards=4, n_test_shards=2)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)
    pairs = select_spatial_idx(val_data, k=args.k)
    print(f"공간 변별쌍 {len(pairs)}개", flush=True)

    def pose_shift(model_):
        """각 쌍에서 이미지 A→B swap 시 예측 pose 변화량(평균 |Δ|) 반환."""
        out = []
        for i, j in pairs:
            fa, pa, _ = val_ex[i]; _, pb, _ = val_ex[j]
            a = predict_continuous(model_, fa, pa, vocab_size, norm)
            b = predict_continuous(model_, fa, pb, vocab_size, norm)
            out.append(float(np.abs(a - b)[POSE].mean()))
        return np.array(out)

    # frozen
    model.projector = orig_base; model.requires_grad_(False); model.eval()
    sh_frozen = pose_shift(model)
    # multiscale_attn
    print("train multiscale_attn…", flush=True)
    model.projector = orig_base; model.requires_grad_(False)
    torch.manual_seed(0)
    proj, params, _ = build_projector("multiscale_attn", model)
    model.projector = proj
    for p in params: p.requires_grad = True
    train(model, params, train_ex, args.steps, args.lr, seed=0)
    model.eval()
    sh_attn = pose_shift(model)

    # GT pose shift (이미지가 실제로 요구하는 변화)
    gt = []
    for i, j in pairs:
        gi = decode_tokens(val_ex[i][2].numpy(), vocab_size, *norm)
        gj = decode_tokens(val_ex[j][2].numpy(), vocab_size, *norm)
        gt.append(float(np.abs(gi - gj)[POSE].mean()))
    gt = np.array(gt)

    # ---- figure: 각 쌍별로 imgA,imgB + pose shift 막대 ----
    k = len(pairs)
    fig, axes = plt.subplots(k, 3, figsize=(9.5, 2.5 * k),
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})
    if k == 1: axes = axes[None, :]
    for r, (i, j) in enumerate(pairs):
        instr = val_data[i][1][:38]
        axes[r, 0].imshow(val_data[i][0]); axes[r, 0].axis("off")
        axes[r, 1].imshow(val_data[j][0]); axes[r, 1].axis("off")
        if r == 0:
            axes[r, 0].set_title("image A (prompt source)", fontsize=9)
            axes[r, 1].set_title("image B (spatially different)", fontsize=9)
        axes[r, 0].text(0, -8, instr, fontsize=7, color="#444")
        ax = axes[r, 2]
        vals = [sh_frozen[r], sh_attn[r], gt[r]]
        cols = ["#9aa0a6", "#d93025", "#188038"]
        ax.barh(["frozen", "multiscale_attn", "GT change"], vals, color=cols, alpha=0.9, edgecolor="black")
        ax.set_xlim(0, max(gt.max(), sh_attn.max(), sh_frozen.max()) * 1.15 + 1e-6)
        for yi, v in enumerate(vals): ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
        if r == 0: ax.set_title("predicted pose shift when A→B (closer to GT = tracks space)", fontsize=8.5)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Spatially-demanding images: does the prediction follow the image change?\n"
                 f"frozen Δ={sh_frozen.mean():.3f}  vs  multiscale_attn Δ={sh_attn.mean():.3f}  "
                 f"(GT Δ={gt.mean():.3f})", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGS, "ms5_spatial_demo.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved: {out}")
    print(f"mean pose shift — frozen {sh_frozen.mean():.4f} | multiscale_attn {sh_attn.mean():.4f} | GT {gt.mean():.4f}")


if __name__ == "__main__":
    main()
