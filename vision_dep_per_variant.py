"""
maxinfo/vision_dep_per_variant.py

Exp 2 — 변종별 인과 vision-dependency 직접 측정 (L1 proxy 아님).

아이디어:
  같은 텍스트 프롬프트로 이미지만 다른-에피소드(=GT 액션이 실제로 다른) 것으로 swap했을 때,
  예측 액션이 얼마나 '따라 움직이나(shift)'를 변종별로 비교한다.
  - vision_shift 큼  = 이미지 변화에 예측이 민감 = 비전정보를 실제로 전달/사용
  - blank_degrade 큼 = 이미지 가리면 예측 무너짐 = 비전 의존성

  각 변종을 in-memory로 학습한 뒤 곧바로 측정(가중치 저장 불필요).

실행:
  python maxinfo/vision_dep_per_variant.py --train_n 1500 --steps 3000 --lr 2e-4 \
      --variants baseline_mlp_frozen mlp_scratch mlp_scratch_ln self_attn honeybee maxinfo
출력: maxinfo/vision_dep_result.json
"""
import os, sys, json, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from data import load_jaco_subset
from projectors_zoo import build_projector
from train_eval import (MODEL_ID, DEVICE, CONFIG_PATH, load_norm, decode_tokens,
                        build_examples, make_batch, train)

OUT = os.path.join(os.path.dirname(__file__), "vision_dep_result.json")


@torch.no_grad()
def predict(model, full, pix, vocab_size, norm):
    q01, q99, mask = norm
    input_ids = full.unsqueeze(0).to(DEVICE)
    out = model(input_ids=input_ids, pixel_values=pix.unsqueeze(0).to(DEVICE))
    pred = out.logits[0, -8:-1, :].argmax(-1).cpu().numpy()
    return decode_tokens(pred, vocab_size, q01, q99, mask)


@torch.no_grad()
def vision_dependency(model, pairs, vocab_size, norm):
    """pairs: [(exA, exB)] — A,B는 GT 액션이 크게 다른 서로 다른 에피소드.
    측정: A프롬프트+A이미지 예측 vs A프롬프트+B이미지 예측의 차이(shift), 가림 악화(degrade)."""
    shifts, degrades = [], []
    for (fa, pa, ta), (fb, pb, tb) in pairs:
        pred_A = predict(model, fa, pa, vocab_size, norm)            # 정상
        pred_Aimg_B = predict(model, fa, pb, vocab_size, norm)       # 텍스트는 A, 이미지만 B로
        pred_blank = predict(model, fa, torch.zeros_like(pa), vocab_size, norm)
        gtA = decode_tokens(ta.numpy(), vocab_size, *norm)
        shifts.append(float(np.abs(pred_A - pred_Aimg_B).mean()))    # 이미지 바뀌면 예측 변화
        degrades.append(float(np.abs(pred_blank - gtA).mean() - np.abs(pred_A - gtA).mean()))
    return float(np.mean(shifts)), float(np.mean(degrades))


def select_pairs(val_ex, val_data, vocab_size, norm, k=12):
    """GT 액션(xyz) 거리가 큰 쌍을 골라 변별력 확보."""
    acts = np.array([d[2] for d in val_data])              # normalized 7D
    n = len(val_ex)
    # 첫 토큰(x)·둘째(y) 기준 정렬해 양 극단을 짝지음
    order = np.argsort(acts[:, 0] + acts[:, 1])
    pairs = []
    for i in range(min(k, n // 2)):
        a = order[i]; b = order[-(i + 1)]
        pairs.append((val_ex[a], val_ex[b]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=1500)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--variants", nargs="+",
                    default=["baseline_mlp_frozen", "mlp_scratch", "mlp_scratch_ln",
                             "self_attn", "honeybee", "maxinfo"])
    args = ap.parse_args()

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}

    print("로드: openvla-7b (4bit)…", flush=True)
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

    train_data, val_data = load_jaco_subset(args.train_n, args.val_n, CONFIG_PATH)
    train_ex = build_examples(train_data, processor, vocab_size)
    val_ex = build_examples(val_data, processor, vocab_size)
    pairs = select_pairs(val_ex, val_data, vocab_size, norm, k=12)
    print(f"vision-dep 평가쌍 {len(pairs)}개 (GT 액션 극단 선별)", flush=True)

    for name in args.variants:
        if name in results:
            print(f"[skip] {name}", flush=True); continue
        print(f"\n===== {name} =====", flush=True)
        model.projector = orig_base
        model.requires_grad_(False)
        proj, params, meta = build_projector(name, model)
        model.projector = proj
        for p in params:
            p.requires_grad = True
        if meta["trainable"] and args.steps > 0:
            train(model, params, train_ex, args.steps, args.lr)
        model.eval()
        shift, degrade = vision_dependency(model, pairs, vocab_size, norm)
        results[name] = dict(vision_shift=shift, blank_degrade=degrade, tokens=meta["tokens"])
        print(f"  vision_shift={shift:.4f} (이미지 바뀌면 예측 변화↑=비전 사용)  "
              f"blank_degrade={degrade:.4f} (가리면 악화↑=비전 의존)", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
        if name != "baseline_mlp_frozen":
            del proj
        model.projector = orig_base
        torch.cuda.empty_cache()

    print(f"\n저장: {OUT}")
    print(f"\n{'variant':<22}{'vision_shift':>14}{'blank_degrade':>15}")
    for k, v in results.items():
        print(f"{k:<22}{v['vision_shift']:>14.4f}{v['blank_degrade']:>15.4f}")


if __name__ == "__main__":
    main()
