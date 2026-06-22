#!/bin/bash
# 교란 분리 실험 자동 체인: Exp1(이미 실행중) 완료 대기 → Exp2 → Exp3
# 각 단계는 자체 이어달리기(JSON에 완료 변종 skip) 지원. 죽으면 재시도.
cd /home/ssu/openvla
source /home/ssu/openvla-env/bin/activate
LOG=/home/ssu/openvla/maxinfo/disentangle.log
VARIANTS="baseline_mlp_frozen mlp_scratch mlp_scratch_ln self_attn honeybee maxinfo"
echo "[chain] 시작 $(date)" >> $LOG

# --- 1) Exp1 (mlp_scratch_ln) 완료 대기 ---
while ! grep -q "저장: maxinfo/compare_real_result" maxinfo/exp1_ln.log 2>/dev/null; do
  if ! pgrep -f "train_eval.py.*mlp_scratch_ln" >/dev/null; then
    if ! grep -q "저장: maxinfo/compare_real_result" maxinfo/exp1_ln.log 2>/dev/null; then
      echo "[chain] Exp1 죽음, 재시작 $(date)" >> $LOG
      nohup python maxinfo/train_eval.py --train_n 1500 --val_n 256 --steps 3000 --lr 2e-4 \
        --variants mlp_scratch_ln >> maxinfo/exp1_ln.log 2>&1 &
      sleep 60
    fi
  fi
  sleep 60
done
echo "[chain] Exp1 완료. 8종 병합 $(date)" >> $LOG
# Exp1 결과를 8종 백업에 병합
python3 -c "
import json
base=json.load(open('maxinfo/compare_real_result.8var_3000.json'))
new=json.load(open('maxinfo/compare_real_result.json'))
base.update({k:v for k,v in new.items() if k=='mlp_scratch_ln'})
json.dump(base, open('maxinfo/compare_real_result.json','w'), indent=2)
print('병합:', list(base.keys()))
" >> $LOG 2>&1

# --- 2) Exp2 vision-dependency (이어달리기) ---
echo "[chain] Exp2 시작 $(date)" >> $LOG
for try in 1 2 3; do
  python maxinfo/vision_dep_per_variant.py --train_n 1500 --steps 3000 --lr 2e-4 \
    --variants $VARIANTS >> maxinfo/disentangle.log 2>&1
  done2=$(python3 -c "import json,os;p='maxinfo/vision_dep_result.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null)
  echo "[chain] Exp2 시도$try 완료 변종 $done2/6" >> $LOG
  [ "$done2" -ge 6 ] && break
  sleep 10
done

# --- 3) Exp3 spatial probe (이어달리기) ---
echo "[chain] Exp3 시작 $(date)" >> $LOG
for try in 1 2 3; do
  python maxinfo/spatial_probe.py --train_n 1500 --steps 3000 --lr 2e-4 \
    --variants $VARIANTS >> maxinfo/disentangle.log 2>&1
  done3=$(python3 -c "import json,os;p='maxinfo/spatial_probe_result.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null)
  echo "[chain] Exp3 시도$try 완료 변종 $done3/6" >> $LOG
  [ "$done3" -ge 6 ] && break
  sleep 10
done

echo "[chain] 전체 완료 $(date)" >> $LOG
touch maxinfo/DISENTANGLE_DONE
