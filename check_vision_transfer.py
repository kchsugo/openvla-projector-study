"""
maxinfo/check_vision_transfer.py

"projector가 비전정보를 실제로 전달하는가"를 직접 확인하는 스크립트.

아이디어:
  비전정보가 잘 전달되면 예측 액션이 '입력 이미지에 의존'해야 한다.
  - (실제 이미지)  예측이 정답(GT)에 가까움  → 비전정보가 흐름
  - (이미지 가림)  예측이 망가짐(GT에서 멀어짐) → 모델이 실제로 이미지를 보고 있었다는 증거
  - (다른 이미지)  예측이 그 이미지 쪽으로 바뀜 → 이미지마다 다르게 반응 = 전달됨

대상: baseline_mlp_frozen (배포된 원본 OpenVLA projector).
      (다른 변종은 학습 가중치를 디스크에 저장하지 않으므로 여기선 원본만 검증)

실행:
  python maxinfo/check_vision_transfer.py --n 8
"""
import os, sys, json, argparse
os.environ["BNB_CUDA_VERSION"] = "130"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from data import load_jaco_subset
from train_eval import (MODEL_ID, DEVICE, CONFIG_PATH, load_norm,
                        decode_tokens, encode_action, build_examples, make_batch)


@torch.no_grad()
def predict_action(model, ex, vocab_size, norm):
    """한 example에 대해 7D 예측 액션(역정규화) 반환."""
    q01, q99, mask = norm
    input_ids, pix, labels, atoks = make_batch(ex)
    out = model(input_ids=input_ids, pixel_values=pix)
    logits = out.logits
    pred = logits[0, -8:-1, :].argmax(-1).cpu().numpy()
    return decode_tokens(pred, vocab_size, q01, q99, mask), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="검증 샘플 수")
    args = ap.parse_args()

    print("로드: openvla-7b (4bit, 원본 projector)…", flush=True)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=["projector"])
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, quantization_config=qc, device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    model.eval()
    vocab_size = model.vocab_size
    norm = load_norm()

    _, val = load_jaco_subset(100, args.n + 1, CONFIG_PATH)
    val = val[:args.n + 1]
    ex = build_examples(val, processor, vocab_size)

    real_l1, blank_l1, other_l1, shift = [], [], [], []
    print(f"\n{'='*78}\n샘플별 검증 (원본 projector, 비전정보 전달 테스트)\n{'='*78}")
    for i in range(args.n):
        full, pix, atoks = ex[i]
        q01, q99, mask = norm
        gt = decode_tokens(atoks.numpy(), vocab_size, q01, q99, mask)

        # ① 실제 이미지
        pa_real, _ = predict_action(model, ex[i], vocab_size, norm)
        # ② 이미지 가림 (0으로)
        blank = (full, torch.zeros_like(pix), atoks)
        pa_blank, _ = predict_action(model, blank, vocab_size, norm)
        # ③ 다른 이미지 (다음 샘플의 픽셀로 교체, 텍스트는 그대로)
        other = (full, ex[i + 1][1], atoks)
        pa_other, _ = predict_action(model, other, vocab_size, norm)

        l1r = np.abs(pa_real - gt).mean()
        l1b = np.abs(pa_blank - gt).mean()
        l1o = np.abs(pa_other - gt).mean()
        sh = np.abs(pa_real - pa_other).mean()  # 이미지 바뀌면 예측이 얼마나 변하나
        real_l1.append(l1r); blank_l1.append(l1b); other_l1.append(l1o); shift.append(sh)

        print(f"[{i}] GT(xyz)={gt[:3].round(3)}  실제예측={pa_real[:3].round(3)}")
        print(f"     L1(실제)={l1r:.4f}  L1(가림)={l1b:.4f}  L1(다른img)={l1o:.4f}  Δ예측(img교체)={sh:.4f}")

    print(f"\n{'='*78}\n요약 (n={args.n})\n{'='*78}")
    print(f"  평균 L1 (실제 이미지) : {np.mean(real_l1):.4f}   ← 작을수록 비전정보 잘 전달")
    print(f"  평균 L1 (이미지 가림) : {np.mean(blank_l1):.4f}   ← 커지면 '이미지를 실제로 봤다'는 증거")
    print(f"  평균 L1 (다른 이미지) : {np.mean(other_l1):.4f}")
    print(f"  평균 Δ예측(이미지 교체): {np.mean(shift):.4f}   ← 0이 아니면 예측이 이미지에 의존")
    deg = (np.mean(blank_l1) - np.mean(real_l1))
    print(f"\n  판정: 이미지를 가리면 L1이 {deg:+.4f} 만큼 악화"
          f"  → {'✅ 비전정보가 실제로 전달되고 있음' if deg > 0.005 else '⚠️ 비전 의존도 약함'}")

    out = dict(n=args.n, mean_l1_real=float(np.mean(real_l1)),
               mean_l1_blank=float(np.mean(blank_l1)),
               mean_l1_other=float(np.mean(other_l1)),
               mean_pred_shift=float(np.mean(shift)),
               degradation_when_blanked=float(deg))
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
              "vision_transfer_result.json"), "w"), indent=2)
    print("\n저장: maxinfo/vision_transfer_result.json")


if __name__ == "__main__":
    main()
