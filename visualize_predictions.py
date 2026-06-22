"""
실제 이미지를 입력받아 6종 모델의 예측 액션을 시각화하는 스크립트
"""
import json
import torch
import numpy as np
from pathlib import Path
import sys

# 데이터 로드
sys.path.insert(0, str(Path(__file__).parent.parent))
from maxinfo.data import load_jaco_subset, _normalizer
from maxinfo.projectors_zoo import build_projector
from openvla.core import OpenVLAChatInterface


def load_models_and_data():
    """모델과 데이터 로드"""
    print("🔧 모델 로드 중...")
    model = OpenVLAChatInterface.load_model(
        "openvla-7b-jaco", hf_token=None, quantized=True
    )

    print("📦 데이터 로드 중...")
    examples = load_jaco_subset(n_train=100, n_val=32)
    norm = _normalizer(Path(__file__).parent.parent / "config")

    return model, examples, norm


def predict_with_variant(model, image, instruction, variant_name):
    """특정 변종으로 액션 예측"""
    projector, _, _ = build_projector(variant_name, model)

    # 이미지 전처리 (OpenVLA 방식)
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", trust_remote_code=True
    )

    image_pil = image if hasattr(image, 'mode') else None
    if image_pil is None:
        from PIL import Image
        image_pil = Image.fromarray((image * 255).astype(np.uint8))

    inputs = processor(images=image_pil, return_tensors="pt")["pixel_values"]

    # Vision 특징 추출
    with torch.no_grad():
        vision_features = model.vision_transformer(
            inputs.to(model.device)
        )  # [1, 256, 2176]

        # Projector 통과
        projected = projector(vision_features)  # [1, 256, 4096]

        # 간단한 평균 풀링 → 액션 토큰 생성
        action_tokens = projected.mean(dim=1)  # [1, 4096]

        # 토큰 → 액션 디코딩 (ActionTokenizer)
        action_logits = model.llm_tokenizer.convert_logits_to_action(
            action_tokens, action_scale=1.0
        )  # [1, 7, 256]

        # argmax → 액션 값
        action_indices = action_logits.argmax(dim=-1)  # [1, 7]
        action_normalized = (action_indices.float() / 128.0 - 1.0).cpu().numpy()[0]

    return action_normalized


def visualize_sample(model, examples, norm, sample_idx=0):
    """샘플 이미지로 6종 예측 비교"""
    sample = examples[sample_idx]
    image = sample['image']  # [224, 224, 3], uint8
    instruction = sample['instruction']
    actual_action = sample['action']  # [7]

    print(f"\n{'='*80}")
    print(f"📸 샘플 #{sample_idx}")
    print(f"{'='*80}")
    print(f"📝 지시: {instruction}")
    print(f"🎯 실제 액션: {actual_action}")
    print(f"\n{'변종명':<20} {'예측 액션':<30} {'L1 오차':<10}")
    print(f"{'-'*80}")

    variants = [
        "baseline_mlp_frozen",
        "baseline_mlp_trained",
        "honeybee",
        "self_attn",
        "cross_attn",
        "maxinfo",
    ]

    results = []
    for variant in variants:
        try:
            pred_action = predict_with_variant(model, image, instruction, variant)
            l1_error = np.abs(pred_action - actual_action).mean()

            action_str = ", ".join([f"{v:+.2f}" for v in pred_action])
            print(f"{variant:<20} [{action_str}] {l1_error:.4f}")
            results.append((variant, pred_action, l1_error))
        except Exception as e:
            print(f"{variant:<20} ERROR: {str(e)[:50]}")

    print(f"{'-'*80}")

    # 최고/최악 성능
    if results:
        best = min(results, key=lambda x: x[2])
        worst = max(results, key=lambda x: x[2])
        print(f"\n✅ 최고 성능: {best[0]} (L1={best[2]:.4f})")
        print(f"❌ 최악 성능: {worst[0]} (L1={worst[2]:.4f})")

    return results


def compare_multiple_samples(model, examples, norm, num_samples=5):
    """여러 샘플 비교"""
    print(f"\n\n{'='*80}")
    print(f"🔍 {num_samples}개 샘플 평균 성능")
    print(f"{'='*80}\n")

    variants = [
        "baseline_mlp_frozen",
        "baseline_mlp_trained",
        "honeybee",
        "self_attn",
        "cross_attn",
        "maxinfo",
    ]

    stats = {v: [] for v in variants}

    for i in range(min(num_samples, len(examples))):
        sample = examples[i]
        image = sample['image']
        instruction = sample['instruction']
        actual_action = sample['action']

        for variant in variants:
            try:
                pred_action = predict_with_variant(model, image, instruction, variant)
                l1_error = np.abs(pred_action - actual_action).mean()
                stats[variant].append(l1_error)
            except:
                pass

    # 평균 오차 출력
    print(f"{'변종':<20} {'평균 L1 오차':<15} {'std':<10}")
    print(f"{'-'*60}")

    sorted_variants = sorted(
        stats.items(),
        key=lambda x: np.mean(x[1]) if x[1] else float('inf')
    )

    for variant, errors in sorted_variants:
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            print(f"{variant:<20} {mean_error:.4f}±{std_error:.4f}")


if __name__ == "__main__":
    try:
        model, examples, norm = load_models_and_data()

        # 첫 번째 샘플 상세 분석
        visualize_sample(model, examples, norm, sample_idx=0)

        # 여러 샘플 비교
        compare_multiple_samples(model, examples, norm, num_samples=5)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
