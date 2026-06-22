#!/bin/bash

# maxinfo/run_and_analyze.sh
# 6종 변종을 학습하고 결과를 분석하는 완전한 가이드

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║           🚀 6종 OpenVLA Projector 직접 실행 및 결과 분석 가이드              ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 0: 환경 설정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd /home/ssu/openvla
source openvla-env/bin/activate

확인:
  python --version        # Python 3.10 이상
  nvidia-smi              # GPU 메모리 (8GB RTX 5060 필요)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 1: 상세 설명 보기 (코드 실행 없음)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd /home/ssu/openvla

# 6종 변종의 상세한 비교 설명 보기 (가장 추천!)
python maxinfo/detailed_comparison_guide.py | less

또는

python maxinfo/detailed_comparison_guide.py  # 전체 출력


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 2: 빠른 smoke 테스트 (5분, 12 step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 코드가 정상 작동하는지 빠르게 확인 (데이터 로드부터 학습까지)
python maxinfo/train_eval.py --smoke

출력 예:
  로드: openvla-7b (4bit)…
  vocab_size=32000
  데이터 로드: train=40 val=16
  ===== baseline_mlp_frozen =====
    acc=0.714 L1=0.0161 ...
  ===== baseline_mlp_trained =====
    step 4/12 loss=1.2134
    step 8/12 loss=1.0876
    acc=0.723 L1=0.0186 ...
  ... (나머지 4종)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 3: 본 실행 (1시간, 800 step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 옵션 A: 백그라운드 실행 (추천 - 다른 작업 가능)
nohup python maxinfo/train_eval.py \
    --train_n 1500 --val_n 256 --steps 800 --lr 2e-4 \
    > maxinfo/train.log 2>&1 &

echo "백그라운드 실행 시작됨. 다음 터미널에서 모니터:"
echo "  tail -f maxinfo/train.log"


# 옵션 B: 포그라운드 실행 (직접 보기, 터미널 점유)
python maxinfo/train_eval.py \
    --train_n 1500 --val_n 256 --steps 800 --lr 2e-4


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 4: 실시간 로그 모니터링 (백그라운드 실행 시)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 터미널 2 에서 실시간 로그 보기
tail -f maxinfo/train.log

# 또는 마지막 50줄만 보기
tail -50 maxinfo/train.log

주요 로그:
  ===== baseline_mlp_frozen =====
    acc=0.714 L1=0.0161 ...   ← 완료

  ===== baseline_mlp_trained =====
    step 133/800 loss=1.3222
    step 266/800 loss=1.0603
    ...
    step 798/800 loss=0.7664 ← 진행 중
    acc=0.751 L1=0.0657 ...   ← 완료

  ===== honeybee =====
    ...

  ===== self_attn =====
    ...

  ===== cross_attn =====
    ...

  ===== maxinfo =====
    ...

  저장: maxinfo/compare_real_result.json ← 완료 신호!


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 5: 결과 확인 (6종 비교표)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

완료 후 결과 JSON 보기:

# 방법 1: JSON 파일 직접 보기 (포맷팅)
cat maxinfo/compare_real_result.json | python -m json.tool

# 방법 2: Python으로 비교표 생성
python << 'PYEOF'
import json
import pandas as pd

with open('maxinfo/compare_real_result.json') as f:
    results = json.load(f)

df = pd.DataFrame(results).T

# 시각적 비교표
print("\n" + "="*100)
print("6종 변종 비교 (jaco_play 실데이터)")
print("="*100 + "\n")

cols = ['token_acc', 'action_l1', 'action_mse', 'trainable_params', 'tokens', 'peak_vram_gb']
display_df = df[cols].copy()
display_df['trainable_params'] = (display_df['trainable_params'] / 1e6).round(1)
display_df.columns = ['토큰정확도', '액션L1오차', '액션MSE', '파라미터(M)', '토큰수', 'VRAM(GB)']

print(display_df.to_string())

print("\n" + "="*100)
print("분석:")
print("="*100)

# 최고/최악 성능
best_l1 = df['action_l1'].idxmin()
worst_l1 = df['action_l1'].idxmax()
best_acc = df['token_acc'].idxmax()
worst_acc = df['token_acc'].idxmin()

print(f"\n🏆 최고 성능 (L1 오차 최소):")
print(f"  {best_l1:<20} L1={df.loc[best_l1, 'action_l1']:.4f}")

print(f"\n❌ 최악 성능 (L1 오차 최대):")
print(f"  {worst_l1:<20} L1={df.loc[worst_l1, 'action_l1']:.4f}")

print(f"\n📊 토큰 정확도 순위:")
for name in df['token_acc'].sort_values(ascending=False).index:
    acc = df.loc[name, 'token_acc']
    err = df.loc[name, 'action_l1']
    print(f"  {name:<20} acc={acc:.3f}  L1={err:.4f}")

print("\n" + "="*100)
PYEOF


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 6: 샘플 이미지로 실제 예측 보기
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 실제 jaco_play 데이터와 토큰화 과정 보기
python maxinfo/show_prediction_example.py

출력 예:
  ════════════════════════════════════════════════╗
  📊 학습 결과 요약
  ════════════════════════════════════════════════

  변종                  토큰정확도      액션L1오차
  ──────────────────────────────────────────────
  baseline_mlp_frozen    0.714 [████████…..]  0.0161
  baseline_mlp_trained   0.723 [████████…..]  0.0186
  honeybee               0.714 [████████…..]  0.0441
  self_attn              0.000 [░░░░░░░░░░░░]  0.2108
  cross_attn             0.152 [██░░░░░░░░░░]  0.1782
  maxinfo                0.714 [████████…..]  0.0161

  ✅ 최고 성능: baseline_mlp_frozen (L1=0.0161)
  ❌ 최악 성능: self_attn (L1=0.2108)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 7: 6종 변종 상세 구조 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 각 변종의 역할과 보완점을 자세히 설명 (Step 1과 동일)
python maxinfo/detailed_comparison_guide.py > comparison_report.txt
less comparison_report.txt


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 전체 흐름 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 상세 설명 읽기 (5분)
   → python maxinfo/detailed_comparison_guide.py

2. 빠른 테스트 (5분)
   → python maxinfo/train_eval.py --smoke

3. 본 실행 선택:
   a) 빠른 테스트 후 결과 (10분)
   b) 백그라운드 본 실행 (1시간)
   c) 포그라운드 본 실행 (1시간, Terminal 점유)

4. 결과 확인:
   → cat maxinfo/compare_real_result.json | python -m json.tool

5. 샘플 분석:
   → python maxinfo/show_prediction_example.py


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 핵심 결론 (이미 알려진 결과)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 변종                   액션L1오차   의미
 ─────────────────────────────────────
 1️⃣ baseline_mlp_frozen 0.0161    원본 최고 성능 (정보 100% 보존) ⭐
 2️⃣ maxinfo             0.0161    residual 구조로 동일 성능 유지 ✅
 3️⃣ baseline_trained    0.0186    재학습하면 오버피팅 (현상 유지)
 4️⃣ honeybee            0.0441    압축하면 정보 손실 (2.7배)
 5️⃣ cross_attn          0.1782    64 query로 정보 미흡 (11배)
 6️⃣ self_attn           0.2108    학습 불안정 (13배, 0% 정확도) ❌

결론:
  ✅ 시각정보 보존 최고: baseline_mlp_frozen (원본)
  ✅ 정보 보존의 실제 의미: "사전학습된 MLP가 이미 최적화됨"
  ✅ enhancement는 더 큰 데이터/복잡 태스크에서 필요


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ FAQ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1: 실행에 얼마나 걸리나요?
A: • smoke 테스트: ~5분 (12 step)
   • 본 실행: ~1시간 (800 step × 6종)

Q2: GPU 메모리는?
A: RTX 5060(8GB)에 최적화됨 (VRAM 4.8~6.3GB 사용)

Q3: 결과 파일은?
A: maxinfo/compare_real_result.json
   - 6종 변종의 성능 메트릭 저장
   - JSON 형식 (pandas/Python으로 분석 가능)

Q4: 각 변종의 코드는?
A: maxinfo/projectors_zoo.py
   - build_projector(name, model) 함수로 생성
   - 각 변종별 구조 정의

Q5: 데이터는?
A: OpenX jaco_play (로컬 TFDS 로드)
   - train: 1500 샘플
   - val: 256 샘플
   - 자동 다운로드 및 캐싱

Q6: 재현 가능성?
A: seed=0 고정
   - 동일 하드웨어에서 동일 결과
   - 결과는 이미 저장됨 (compare_real_result.json)


EOF
