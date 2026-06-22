#!/bin/bash
# scaling 실험 자동 감시·재시작 watchdog
# 20/20 완료될 때까지: 프로세스 죽으면 이어달리기로 재시작
cd /home/ssu/openvla
source /home/ssu/openvla-env/bin/activate

CMD="python /home/ssu/openvla/maxinfo/scaling_curve.py --sizes 500 2000 5000 10000 30000 --variants baseline_mlp_frozen mlp_scratch self_attn honeybee --steps_per_epoch 2 --lr 2e-5 --val_n 256"
LOG=/home/ssu/openvla/maxinfo/scaling.log
JSON=/home/ssu/openvla/maxinfo/scaling_result.json
WLOG=/home/ssu/openvla/maxinfo/watchdog.log

echo "[watchdog] 시작 $(date)" >> $WLOG
while true; do
  # 완료 셀 수 확인
  done=$(python3 -c "import json;print(len(json.load(open('$JSON'))))" 2>/dev/null || echo 0)
  if [ "$done" -ge 20 ]; then
    echo "[watchdog] 20/20 완료, 종료 $(date)" >> $WLOG
    break
  fi
  # scaling 프로세스 살아있나?
  alive=$(pgrep -f "scaling_curve.py" | wc -l)
  if [ "$alive" -eq 0 ]; then
    echo "[watchdog] $(date) 프로세스 죽음(완료 $done/20). 재시작." >> $WLOG
    nohup $CMD >> $LOG 2>&1 &
    sleep 60   # 모델 로드 대기
  fi
  sleep 120
done
echo "[watchdog] 완전 종료 $(date)" >> $WLOG
