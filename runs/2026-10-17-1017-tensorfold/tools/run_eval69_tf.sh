#!/bin/bash
# 69-scenario tool suite against the TensorFold lane (thinking off), same command shape as the vLLM 1017g run.
# usage: run_eval69_tf.sh "<label>" "<note>"   (on the Mac mini, which holds 2wild-model-eval; EP is an ssh tunnel to the lane's :8001)
OUT=~/.openclaw/workspace/2wild-model-eval-results/dsv41-1017-tensorfold
mkdir -p $OUT
EP=${EP:-http://127.0.0.1:18001/v1}; M=deepseek-v4.1-flash
echo "=== $(date) 69 eval start ==="
cd ~/.openclaw/workspace/2wild-model-eval && EVAL_THINKING=off PYTHONUNBUFFERED=1 python3 run_eval.py --endpoint $EP --model $M \
  --label "$1" \
  --quant "EXL3 3.5 bpw, all matrices (LibertAI), TensorFold native engine" --cluster "4x DGX Spark GB10 TP4 (Reddie head), TensorFold" --ctx "1,048,576" \
  --note "$2" \
  --timeout 300 > $OUT/run_eval.log 2>&1; echo "=== $(date) 69 eval rc=$? ==="
