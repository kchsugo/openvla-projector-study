"""
maxinfo/mlp_bottleneck_analysis.py

2-Layer MLP의 진정한 단점과 해결 방안
"""

def analysis():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║           🔍 2-Layer MLP의 진정한 단점과 해결 방안                             ║
║                                                                                ║
║        Vision (2176dim) ←→ MLP ←→ LLM (4096dim)                               ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 1단계: 2-Layer MLP 구조 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현재 구조:

입력: Vision Features
├─ 크기: [Batch, 256 tokens, 2176 dim]
├─ 출처: DINOv2 + SigLIP 합침
└─ 특징: 통합된 visual representation

    ↓ (Token-wise forward)

Layer 1:
├─ Linear(2176 → 4096)
├─ ReLU
└─ Dropout(0.1)

    ↓

Layer 2:
├─ Linear(4096 → 4096)
├─ ReLU
└─ Dropout(0.1)

    ↓

Layer 3:
├─ Linear(4096 → 4096)
└─ (no activation)

    ↓

출력: LLM Features
├─ 크기: [Batch, 256 tokens, 4096 dim]
└─ 사용처: LLM 입력 임베딩


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 2단계: 2-Layer MLP의 실제 단점들
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 단점 1: "독립적 처리" (No Cross-token Communication)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현상:
  각 토큰이 독립적으로 처리됨
  256 토큰 × 3 Linear layer
  But NO interaction between tokens

문제:
  ┌─────────────┐
  │ Token 1     │  → Linear chains → [4096]
  └─────────────┘

  ┌─────────────┐
  │ Token 2     │  → Linear chains → [4096]
  └─────────────┘
              (no cross-talk)
  ┌─────────────┐
  │ Token 256   │  → Linear chains → [4096]
  └─────────────┘

실제 의미:
  • 왼쪽 이미지와 오른쪽 이미지 정보가 섞이지 않음
  • 객체들 간의 관계 정보 손실
  • "이것과 저것의 관계"를 표현 못 함

예시:
  "빨간 공이 파란 상자 위에 있다"
  → MLP는 "빨간 공의 특징"과 "파란 상자의 특징"을
    별도로 처리함 (관계 미반영)

우리가 시도한 개선:
  ✓ self_attn: 토큰 간 상호작용 추가
  ✗ 결과: 실패 (L1=0.2108, 최악)

왜 실패했나?
  1. Attention 가중치 랜덤 초기화
  2. 사전학습 정렬 파괴
  3. 적은 데이터(1500)로는 학습 불안정


❌ 단점 2: "Dimensionality Mismatch" (차원 불균형)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

입출력 차원:

Vision Input:
  └─ Each token: 2176 dim

Vision → 4096:
  └─ Information compression/expansion
  └─ 정보 손실 또는 과잉 생성?

문제 분석:

  2176 → 4096: Information explosion
  ┌────────────────────────────────┐
  │ 원본 차원: 2176                  │
  │ 확장 후: 4096 (1.88배)          │
  │                                │
  │ 질문: 실제 정보 있나?          │
  │ • 사전학습으로 채워짐 (가능성 1) │
  │ • 의미 없는 패딩 (가능성 2)     │
  └────────────────────────────────┘

실제로 필요한 차원? 불명
  • Information theory:
    2176 dim이면 충분할 수 있음
  • But LLM embedding은 4096 dim
  • Mismatch 최소화를 위해 확장

우리가 시도한 개선:
  ✓ honeybee: 256 → 64 토큰으로 압축
  ✗ 결과: 실패 (L1=0.0441, 정보 손실)

왜 실패했나?
  • 4개 토큰 정보가 1개로 압축됨
  • 공간 정보 손실 (이웃 토큰 혼합)


❌ 단점 3: "Non-linearity 약함" (Expressiveness 제한)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현재 구조:
  Layer 1: Linear(2176→4096) → ReLU → Dropout
  Layer 2: Linear(4096→4096) → ReLU → Dropout
  Layer 3: Linear(4096→4096) (no activation)

문제:
  3개 Linear layer + 2개 ReLU만으로 충분한가?

분석:
  Vision (2176) → 4096
  이 변환이 얼마나 복잡할까?

  예상:
    • Simple projection이면 Linear 1개로 충분
    • 하지만 현실:
      - Vision encoder와 LLM embedding이 매우 다들 수 있음
      - Cross-modal alignment가 비선형적일 수 있음
      - 더 많은 비선형성 필요할 수 있음

우리가 시도한 개선:
  ✓ self_attn: 비선형성 추가 (attention mechanism)
  ✗ 결과: 실패 (초기화 문제)
  ✓ maxinfo: 더 깊은 구조 추가
  ✗ 결과: gamma=0 (불필요함)

왜 실패했나?
  • jaco_play 데이터는 단순함
  • 현재 MLP + 사전학습으로 충분
  • 더 복잡한 데이터에서는 필요할 수 있음


❌ 단점 4: "Context Window 부족" (Long-range Dependencies)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현상:
  각 토큰이 256 dim으로만 대표됨
  정보 병목 가능성

문제:

  Vision Information Density:
  ├─ 원본 이미지: 224×224×3 = 150,528 pixels
  ├─ Vision encoder: 256 tokens × 2176 dim = 557,056 features
  ├─ MLP output: 256 tokens × 4096 dim = 1,048,576 features
  └─ 정보 밀도 손실: 150,528 → 1,048,576 (팽창했지만 새 정보 없음)

의미:
  원본 이미지의 150K 픽셀 정보
  → 257K 차원의 벡터로 압축 (사전학습)
  → 1M 차원의 벡터로 확장 (MLP, 하지만 새 정보 없음)

  결과: "정보 재배열일 뿐, 새 정보 생성 X"


❌ 단점 5: "Adaptive Mechanism 없음" (No Conditional Processing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현상:
  MLP는 모든 입력에 동일한 변환 적용
  조건부 처리 불가

문제:

  시나리오:
    "이미지 1: 개와 수행기 있음" → 액션 A
    "이미지 2: 개와 고양이 있음" → 액션 B
    (같은 이미지일 수도, 다른 이미지일 수도)

  MLP는?
    image → [same linear transformation]
    → 조건부 로직 표현 불가

  필요한 것:
    image + condition → [adaptive transformation]

현재 해결책:
  • Language instruction "handle the dog differently"
  • LLM이 이를 Action으로 변환
  • But MLP는 여기에 참여 X (정적 변환)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 3단계: 2-Layer MLP의 한계가 "지금 안 드러나는 이유"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

우리 데이터(jaco_play):
  ✅ 단순한 작업 (pick-and-place)
  ✅ 명확한 입출력 대응 (1:1 매핑)
  ✅ 다양한 상황 적을 (제한된 환경)
  ✅ 선형적 관계 (거리↑ → 비선형적 움직임 낮음)

이런 특징들 때문에:
  • 토큰 간 상호작용 필요 없음 (각 픽셀 독립적)
  • 비선형성 적음 (단순 선형 변환으로 충분)
  • 조건부 로직 없음 (같은 명령 반복)
  • 정보 병목 없음 (충분한 차원)

→ 2-layer MLP로 충분!


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 4단계: 2-Layer MLP를 개선하는 방법 (실제 해결책)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 방법 1: "Frozen Pre-training 유지 + 다른 곳 개선"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

가장 실용적 접근:

MLP는 그대로 두고, 다른 곳 개선:
  ✓ Vision Backbone 개선 (더 나은 encoder)
  ✓ Language prompt 최적화 (더 좋은 지시)
  ✓ Action tokenizer 개선 (더 세밀한 제어)
  ✓ LLM 파인튜닝 (액션 토큰 예측 개선)

왜?
  • MLP는 이미 사전학습되어 좋음
  • 다른 컴포넌트가 병목일 수 있음
  • MLP 재설계의 리스크 큼

비용:
  • 낮음 (기존 구조 활용)

효과:
  • 중간 정도 (각 부분 개선의 합)


✅ 방법 2: "Residual-based Adapter"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

개선된 구조:

[Vision 2176] → [Base MLP (frozen)] → [4096]
                       ↑
                       └─ [Enhancement adapter]
                           ├─ Bottle-neck (1024 dim)
                           ├─ 1-2 self-attn blocks
                           └─ Output 4096

특징:
  • Base (frozen)는 유지
  • Enhancement는 작고 트레이닝 가능
  • Residual 연결로 안정성

이우리가 시도한 maxinfo와 비슷하지만:
  • 더 작은 병목 (2048 → 1024)
  • 더 신중한 초기화 (기존 가중치 기반)
  • Pre-trained warm-start

효과:
  • 중간 정도 (혼합 신호)

문제:
  • 현재 데이터에는 불필요 (gamma=0)
  • 큰 데이터에서 테스트 필요


✅ 방법 3: "Conditional MLP" (실제 해결책!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

조건부 처리 추가:

입력: [Vision 2176] + [Instruction embedding]
      (기존)           (새로운)

변환:
  inst_embed = encode_instruction(text)  # 256 dim
  guidance = Linear(256 → 4096)          # 가이던스

구조:
  base_output = MLP(vision)              # [4096]
  guidance = get_guidance(instruction)   # [4096]

  final = base_output + α × guidance

특징:
  • MLP는 frozen 유지
  • 경량 guidance adapter만 학습
  • Instruction 기반 조정

언제 도움?
  ✓ 같은 상황, 다른 명령 → 다른 액션
  ✓ 예: "빨간 공 집기" vs "파란 공 집기"

효과:
  • 높음 (조건부 로직)

비용:
  • 낮음 (경량 adapter)


✅ 방법 4: "Hierarchical Projector" (구조적 개선)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

더 나은 아키텍처:

입력: [Vision 2176] (256 tokens)

Step 1: Local aggregation (각 토큰 근처와 상호작용)
  └─ Window-based attention (3×3 window)
  └─ Output: 256 tokens × 2176 dim

Step 2: Global projection
  └─ Linear(2176 → 4096)
  └─ ReLU
  └─ Output: 256 tokens × 4096 dim

특징:
  • 토큰 간 상호작용 (인접 토큰만)
  • 경량 (full attention X)
  • 공간 정보 보존

왜?
  • Full attention (256²)은 비싸고 불안정
  • 국소 정보만으로도 충분할 가능성
  • 자연 이미지는 국소성 있음 (inductive bias)

효과:
  • 중간~높음 (데이터 특성에 따라)

비용:
  • 중간 (약간의 추가 계산)


✅ 방법 5: "Knowledge Distillation" (지식 증류)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

더 나은 사전학습:

Step 1: 더 큰 모델 학습 (Teacher)
  └─ Deeper MLP 또는 Attention-based
  └─ 1M+ trajectory로 학습
  └─ 매우 좋은 성능 달성

Step 2: Knowledge 추출
  └─ Intermediate activations 캡처
  └─ 학습된 표현 분석

Step 3: 작은 2-layer MLP 학습
  └─ Teacher output 모방
  └─ Distillation loss

결과:
  • 2-layer에 teacher의 지식 압축
  • frozen + distelled = 더 나은 성능

효과:
  • 높음 (사전학습 품질 향상)

비용:
  • 높음 (teacher 모델 훈련 필요)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 5단계: 방법별 비교
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

방법                   예상효과  비용   구현난도  지금 데이터에서
────────────────────────────────────────────────────────────
1. 다른곳 개선         중간       낮    낮      도움 가능
2. Residual adapter    중간       낮    중      거의 도움 X
3. Conditional MLP     높음       낮    중      도움 가능
4. Hierarchical        중간       중    중      도움 가능
5. Knowledge distill   높음       높    높      도움 가능


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 6단계: 결론 및 권장사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2-Layer MLP의 단점 정리:

1️⃣ 토큰 간 상호작용 없음
   → 현재: 단점 아님 (jaco_play는 단순함)
   → 미래: 도움 가능 (복잡한 데이터)

2️⃣ 비선형성 약함
   → 현재: 충분함 (사전학습 + 3 ReLU)
   → 미래: 더 필요할 수 있음

3️⃣ 조건부 처리 불가
   → 현재: 불필요 (명령이 다름, 액션도 다름)
   → 미래: 필요할 수 있음 (같은 명령, 다른 상황)

4️⃣ 정보 병목 없음
   → 현재: 문제 X
   → 미래: 극도로 복잡한 작업에서 보이려나?


실제 권장사항:

지금 해야 할 것 (jaco_play):
  ✅ 1단계: 다른 곳 병목 찾기
     - Vision backbone 성능 측정
     - Language embedding 품질 확인
     - Action tokenizer 정확도 검증

  ✅ 2단계: 다른 곳부터 개선
     - MLP는 그대로 두기
     - Vision encoder 업그레이드
     - Language prompt 최적화

미래 고려사항:
  ✓ 더 큰 데이터 → Residual adapter 시도
  ✓ 조건부 로직 필요 → Conditional MLP
  ✓ 복잡한 환경 → Hierarchical projector
  ✓ 극도로 복잡 → 새로운 사전학습 (distillation)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 최종 핵심 메시지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "2-layer MLP가 너무 단순한데, 단점이 뭐고 어떻게 해결할까?"

A:
  단점:
    1. 토큰 간 상호작용 없음
    2. 비선형성 약할 수 있음
    3. 조건부 처리 불가
    4. 정보 재배열만 (새 정보 X)

  왜 지금 안 드러나나?
    • jaco_play는 단순함
    • 사전학습이 이미 충분함
    • MLP는 단순함이 강점

  해결 방안 (우선순위):
    1순위: "다른 곳" 개선 먼저
           - Vision backbone
           - Language processing
           - Action tokenizer
           → MLP 자체 문제 아닐 수 있음

    2순위: Conditional MLP
           - 경량 guidance adapter
           - 조건부 처리 추가
           - 낮은 비용

    3순위: Hierarchical/Residual
           - 더 큰 데이터 필요
           - 복잡한 작업 필요
           - 지금은 불필요

  결론:
    "2-layer MLP를 고치기 전에,
     정말 MLP가 문제인지 확인하자"

""")

if __name__ == "__main__":
    analysis()
