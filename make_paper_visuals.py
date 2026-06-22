"""
maxinfo/make_paper_visuals.py

논문용 시각화 생성:
  (V1) 8종 projector activation heatmap — 실제 사진을 각 projector가 LLM 공간으로
       사상한 뒤, 토큰(패치)별 활성 강도를 16x16 격자로 복원해 이미지 위에 overlay.
       "어느 projector가 이미지의 어디를 강하게 인코딩하는가"를 보여줌.
  (V2) vision-drop 비교 — 실제 이미지 vs 가린 이미지의 projector 출력 차이.
       projector가 비전정보를 실제로 담고 있음을 시각적으로 증명.

실행:
  python maxinfo/make_paper_visuals.py --image test/my_room.jpg
출력: maxinfo/figs/v1_projector_heatmaps.png, v2_vision_drop.png
"""
import os, sys, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from projectors_zoo import build_projector
from train_eval import MODEL_ID, DEVICE

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

VARIANTS = ["baseline_mlp_frozen", "baseline_mlp_trained", "mlp_scratch", "honeybee",
            "self_attn", "cross_attn", "maxinfo", "maxinfo_scratch"]
TITLE = {"baseline_mlp_frozen": "MLP-frozen (원본)", "baseline_mlp_trained": "MLP-trained",
         "mlp_scratch": "MLP-scratch", "honeybee": "honeybee (64tok)",
         "self_attn": "self-attn", "cross_attn": "cross-attn (64tok)",
         "maxinfo": "maxinfo (제안)", "maxinfo_scratch": "maxinfo-scratch"}


def to_grid(feat):
    """projector 출력 [1, N, D] -> 토큰별 L2 활성 -> sqrt(N)x sqrt(N) 격자 [0,1]."""
    act = feat[0].float().norm(dim=-1).cpu().numpy()   # [N] 토큰별 강도
    n = int(round(len(act) ** 0.5))
    g = act[: n * n].reshape(n, n)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


def upscale(g, size=224):
    t = torch.tensor(g).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False).squeeze().numpy()


@torch.no_grad()
def vision_features(model, pixel_values):
    """OpenVLA 백본에서 projector 입력(=fused patch features) 추출."""
    pv = pixel_values.to(DEVICE, dtype=torch.float16)
    # featurizer가 dino+siglip을 합쳐 fused feature를 만든다
    if hasattr(model, "vision_backbone"):
        return model.vision_backbone(pv)
    raise RuntimeError("vision_backbone 접근 실패")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="test/my_room.jpg")
    args = ap.parse_args()

    print("로드: openvla-7b (4bit)…", flush=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=["projector"])
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, quantization_config=qc, device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    model.eval()
    orig_base = model.projector

    image = Image.open(args.image).convert("RGB")
    disp = image.resize((224, 224))
    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    enc = processor(prompt, image, return_tensors="pt")
    pix = enc["pixel_values"]

    vf = vision_features(model, pix)                       # [1, N, vision_dim]
    print(f"vision feature shape = {tuple(vf.shape)}", flush=True)
    blank_vf = vision_features(model, torch.zeros_like(pix))

    # ---------- V1: 8종 projector heatmap ----------
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    axes = axes.flatten()
    axes[0].imshow(disp); axes[0].set_title("입력 이미지", fontsize=11); axes[0].axis("off")

    for i, name in enumerate(VARIANTS):
        model.projector = orig_base
        proj, _, _ = build_projector(name, model)
        model.projector = proj
        with torch.no_grad():
            out = proj(vf if name == "baseline_mlp_frozen" else vf.to(torch.float16))
        g = upscale(to_grid(out))
        ax = axes[i + 1]
        ax.imshow(disp)
        ax.imshow(g, cmap="jet", alpha=0.55)
        ax.set_title(TITLE[name], fontsize=10)
        ax.axis("off")
        del proj
        torch.cuda.empty_cache()
    model.projector = orig_base
    fig.suptitle("Projector별 시각정보 인코딩 강도 (실제 사진 기반)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIG}/v1_projector_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("저장: figs/v1_projector_heatmaps.png", flush=True)

    # ---------- V2: vision-drop (원본 MLP) ----------
    proj = orig_base
    with torch.no_grad():
        real = to_grid(proj(vf))
        blank = to_grid(proj(blank_vf))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(disp); ax[0].set_title("입력 이미지"); ax[0].axis("off")
    ax[1].imshow(disp); ax[1].imshow(upscale(real), cmap="jet", alpha=0.55)
    ax[1].set_title("실제 이미지 → 인코딩 (정보 풍부)"); ax[1].axis("off")
    ax[2].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
    ax[2].imshow(upscale(blank), cmap="jet", alpha=0.55)
    ax[2].set_title("가린 이미지 → 인코딩 (정보 소실)"); ax[2].axis("off")
    fig.suptitle("Vision-drop: projector가 비전정보를 실제로 전달함을 시각화", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG}/v2_vision_drop.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("저장: figs/v2_vision_drop.png", flush=True)


if __name__ == "__main__":
    main()
