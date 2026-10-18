#!/bin/bash
# 69-scenario tool suite against the live full-EXL3 DeepSeek-V4.1-Flash-1017 TP4 lane (same command shape as the 1017g run, thinking off).
# usage: run_eval69_exl3.sh "<label>" "<ctx>" "<note>"        (run on the Mac mini, which holds 2wild-model-eval)
OUT=~/.openclaw/workspace/2wild-model-eval-results/dsv41-1017-exl3
mkdir -p $OUT
EP=http://<tailnet-ip>:8000/v1; M=deepseek-v4.1-flash
echo "=== $(date) 69 eval start ==="
cd ~/.openclaw/workspace/2wild-model-eval && EVAL_THINKING=off PYTHONUNBUFFERED=1 python3 run_eval.py --endpoint $EP --model $M \
  --label "$1" \
  --quant "EXL3 3.5 bpw, all matrices (LibertAI)" --cluster "4x DGX Spark GB10 TP4 (Reddie head)" --ctx "$2" \
  --note "$3" \
  --timeout 300 > $OUT/run_eval.log 2>&1; echo "=== $(date) 69 eval rc=$? ==="
