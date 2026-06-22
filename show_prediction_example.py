"""
maxinfo/show_prediction_example.py

실제 jaco_play 이미지와 6종 모델의 예측을 비교하는 스크립트
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maxinfo.data import load_jaco_subset
from maxinfo.train_eval import BINS, BIN_CENTERS, decode_tokens, encode_action, load_norm
import numpy as np

CONFIG_PATH = "my_openvla_honeybee/config.json"

def show_sample(sample_idx=0):
    """샘플 데이터 표시"""
    print(f"데이터 로드 중...")
    train_data, val_data = load_jaco_subset(100, 50, CONFIG_PATH)

    # 배치 데이터 병합
    all_data = train_data + val_data
    if sample_idx >= len(all_data):
        print(f"❌ 샘플 인덱스 {sample_idx}는 범위를 벗어났습니다 (max={len(all_data)-1})")
        return

    img, instr, action = all_data[sample_idx]
    print(f"\n{'='*80}")
    print(f"📸 샘플 #{sample_idx}")
    print(f"{'='*80}")
    print(f"📝 지시: {instr}")
    print(f"🤖 실제 액션 (정규화): {action}")
    print(f"   → x={action[0]:+.3f}, y={action[1]:+.3f}, z={action[2]:+.3f}")
    print(f"   → rx={action[3]:+.3f}, ry={action[4]:+.3f}, rz={action[5]:+.3f}, gripper={action[6]:+.3f}")
    print(f"\n📊 이미지: {img.shape} {img.dtype}")
    print(f"   전체 픽셀값 범위: [{img.min()}, {img.max()}]")

    # 토큰화
    norm = load_norm()
    vocab_size = 32000
    tokens = encode_action(action, vocab_size)

    print(f"\n🔢 액션 토큰화 (7개 차원):")
    for dim in range(7):
        t = tokens[dim]
        bin_idx = vocab_size - t - 1
        bin_val = BIN_CENTERS[bin_idx] if bin_idx < len(BIN_CENTERS) else 1.0
        print(f"   dim{dim}: token={t} → bin_idx={bin_idx} → bin_val={bin_val:+.3f}")

    print(f"\n{'='*80}")
    print(f"💡 모델이 하는 일:")
    print(f"   1. 이미지 → Vision 특징 추출 (1x256x2176)")
    print(f"   2. 특징 → Projector 통과 (1x256x4096)")
    print(f"   3. 2176→4096 (비전 정보 확장 또는 압축)")
    print(f"   4. LLM이 마지막 7개 토큰을 액션으로 해석")
    print(f"   5. 각 차원별로 256개 bin 중 예측")
    print(f"\n✅ 완벽한 예측 = 각 차원의 bin이 정확히 일치")

def compare_with_training_results():
    """학습 결과 JSON에서 샘플 예측 표시"""
    result_path = Path(__file__).parent / "compare_real_result.json"

    if not result_path.exists():
        print(f"⚠️  {result_path} 파일이 없습니다.")
        print(f"먼저 다음을 실행하세요:")
        print(f"  python maxinfo/train_eval.py --smoke")
        return

    with open(result_path) as f:
        results = json.load(f)

    print(f"\n\n{'='*80}")
    print(f"📊 학습 결과 요약")
    print(f"{'='*80}\n")

    print(f"{'변종':<20} {'토큰정확도':<12} {'액션L1오차':<12} {'파라미터':<12}")
    print(f"{'-'*80}")

    for variant, metrics in results.items():
        tok_acc = metrics.get('token_acc', 0)
        l1_err = metrics.get('action_l1', 0)
        params = metrics.get('trainable_params', 0)

        # 정확도 시각화 (바그래프)
        bar = "█" * int(tok_acc * 20) + "░" * (20 - int(tok_acc * 20))

        print(f"{variant:<20} {tok_acc:.3f} [{bar}]  {l1_err:.4f}      {params/1e6:>6.1f}M")

    # 최고/최악 성능
    print(f"\n{'='*80}")
    best_l1 = min(results.items(), key=lambda x: x[1]['action_l1'])
    worst_l1 = max(results.items(), key=lambda x: x[1]['action_l1'])
    best_acc = max(results.items(), key=lambda x: x[1]['token_acc'])
    worst_acc = min(results.items(), key=lambda x: x[1]['token_acc'])

    print(f"✅ 액션오차 최고: {best_l1[0]:<20} L1={best_l1[1]['action_l1']:.4f}")
    print(f"❌ 액션오차 최악: {worst_l1[0]:<20} L1={worst_l1[1]['action_l1']:.4f}")
    print(f"✅ 토큰정확도 최고: {best_acc[0]:<20} acc={best_acc[1]['token_acc']:.3f}")
    print(f"❌ 토큰정확도 최악: {worst_acc[0]:<20} acc={worst_acc[1]['token_acc']:.3f}")

    # 의미 설명
    print(f"\n{'='*80}")
    print(f"📖 지표 의미:")
    print(f"  • 토큰정확도: 7개 액션 차원을 256개 bin 중 정확하게 분류한 비율")
    print(f"    - 0.7 = 대부분 맞음 ✅")
    print(f"    - 0.0 = 전혀 못 맞춤 ❌")
    print(f"\n  • 액션L1오차: 예측 액션과 실제 액션의 평균 절대 오차 ([-1,1] 범위)")
    print(f"    - 0.01 = 매우 정확함 ✅ (로봇도 따라할 수 있음)")
    print(f"    - 0.2  = 매우 부정확함 ❌ (로봇 안전 문제 가능)")
    print(f"\n  • 파라미터: 학습 가능한 가중치 수 (많으면 성능 ↑ 하지만 오버피팅 위험)")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="샘플 인덱스 (0~)")
    ap.add_argument("--results", action="store_true", help="학습 결과 요약 보기")
    args = ap.parse_args()

    if args.results or True:  # 항상 결과 표시
        compare_with_training_results()

    if not args.sample < 1000:
        show_sample(args.sample)

if __name__ == "__main__":
    main()
