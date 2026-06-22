#!/usr/bin/env bash
cd /home/ssu/openvla
source /home/ssu/openvla-env/bin/activate
export BNB_CUDA_VERSION=130
echo "[$(date)] start ablation"
python maxinfo/llm_attn_ablation.py --train_n 1500 --steps 800 --val_n 128 > maxinfo/llm_attn_ablation.log 2>&1
echo "[$(date)] start attn_map"
python maxinfo/attn_map_analysis.py --n 8 > maxinfo/attn_map_analysis.log 2>&1
echo "[$(date)] ALL DONE" > maxinfo/CRITIQUE_DONE
