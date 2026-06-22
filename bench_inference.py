"""
maxinfo/bench_inference.py

추론 효율 벤치마크: visual 토큰 수(64 vs 256)에 따른 추론 latency·peak VRAM 측정.
on-device 효율 주장을 위한 그래프 1장용.

- 같은 모델·이미지·프롬프트로 projector만 교체
- 액션 생성(7토큰 greedy decode) latency를 warmup 후 N회 평균
- 토큰 수별 peak VRAM(추론 시점) 측정

실행: python maxinfo/bench_inference.py --image test/my_room.jpg --reps 20
출력: maxinfo/bench_inference_result.json, figs/fig6_inference_efficiency.png
"""
import os, sys, json, time, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
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
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

# 토큰 수가 다른 대표 변종만 (효율은 토큰 수가 좌우)
VARIANTS = ["baseline_mlp_frozen", "honeybee", "cross_attn"]   # 256, 64, 64
TOKENS = {"baseline_mlp_frozen": 256, "honeybee": 64, "cross_attn": 64}


@torch.no_grad()
def time_generate(model, input_ids, pixel_values, reps, warmup=3, new_tokens=7):
    # warmup
    for _ in range(warmup):
        _ = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                           max_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        _ = model.generate(input_ids=input_ids, pixel_values=pixel_values,
                          max_new_tokens=new_tokens, do_sample=False)
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000)  # ms
    peak = torch.cuda.max_memory_allocated() / 1e9
    return float(np.mean(ts)), float(np.std(ts)), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="test/my_room.jpg")
    ap.add_argument("--reps", type=int, default=20)
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
    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    enc = processor(prompt, image, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    pix = enc["pixel_values"].to(DEVICE, dtype=torch.float16)

    results = {}
    for name in VARIANTS:
        model.projector = orig_base
        proj, _, _ = build_projector(name, model)
        model.projector = proj
        mean, std, peak = time_generate(model, input_ids, pix, args.reps)
        results[name] = dict(tokens=TOKENS[name], latency_ms=mean, latency_std=std,
                             peak_vram_gb=peak)
        print(f"  {name:<22} tok={TOKENS[name]:>3}  latency={mean:6.1f}±{std:.1f}ms  VRAM={peak:.2f}GB",
              flush=True)
        if name != "baseline_mlp_frozen":
            del proj
        model.projector = orig_base
        torch.cuda.empty_cache()

    json.dump(results, open(os.path.join(HERE, "bench_inference_result.json"), "w"), indent=2)

    # ---- 그래프: 토큰수별 latency + VRAM ----
    names = list(results.keys())
    labels = [f"{n}\n({results[n]['tokens']}tok)" for n in names]
    lat = [results[n]["latency_ms"] for n in names]
    err = [results[n]["latency_std"] for n in names]
    vram = [results[n]["peak_vram_gb"] for n in names]
    colors = ["#2ca02c" if results[n]["tokens"] == 256 else "#ff7f0e" for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    b = ax[0].bar(labels, lat, yerr=err, color=colors, capsize=4)
    for bi, v in zip(b, lat):
        ax[0].text(bi.get_x()+bi.get_width()/2, v, f"{v:.0f}ms", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylabel("액션 생성 latency (ms, 낮을수록 빠름)")
    ax[0].set_title("추론 속도: 64토큰 vs 256토큰")

    b2 = ax[1].bar(labels, vram, color=colors)
    for bi, v in zip(b2, vram):
        ax[1].text(bi.get_x()+bi.get_width()/2, v, f"{v:.2f}GB", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel("추론 peak VRAM (GB)")
    ax[1].set_title("추론 메모리: 64토큰 vs 256토큰")
    fig.suptitle("On-device 효율: 토큰 압축(64)의 추론 이점", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG}/fig6_inference_efficiency.png", dpi=200, bbox_inches="tight")
    print("\n저장: bench_inference_result.json, figs/fig6_inference_efficiency.png", flush=True)


if __name__ == "__main__":
    main()
