#!/bin/bash
# 69-scenario tool suite against the TP2 lane (thinking off), same command shape as the TP4 run.
# usage: run_eval69_tf_tp2.sh "<label>" "<note>"   (Mac mini / Mac, which hold 2wild-model-eval; EP defaults to the fleet endpoint)
OUT=~/.openclaw/workspace/2wild-model-eval-results/dsv41-1017-tensorfold-tp2
mkdir -p $OUT
EP=${EP:-http://100.113.138.96:8000/v1}; M=deepseek-v4.1-flash
echo "=== $(date) 69 eval start ==="
cd ~/.openclaw/workspace/2wild-model-eval && EVAL_THINKING=off PYTHONUNBUFFERED=1 python3 run_eval.py --endpoint $EP --model $M \
  --label "$1" \
  --quant "EXL3 2.9 bpw, all matrices (LibertAI), TensorFold native engine" --cluster "2x DGX Spark GB10 TP2 (Reddie head + Bluey)" --ctx "1,048,576" \
  --note "$2" \
  --timeout 300 > $OUT/run_eval.log 2>&1; echo "=== $(date) 69 eval rc=$? ==="
