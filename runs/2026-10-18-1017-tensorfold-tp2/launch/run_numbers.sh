#!/bin/bash
# run_numbers.sh (on Reddie, TP2 lane up on :8000): bench C1-C4 + prefill, concurrent check, sampling check, then needles.
B=/var/tmp/tf1017-tp2; cd $B; mkdir -p bench/tp2a
echo "bench start $(date -u +%T)"
python3 tools/v41bench_tf.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label tp2a --out bench/tp2a --levels 1,2,3,4 --prefill 2000,8000,32000,64000,128000 > bench/tp2a/bench.txt 2>&1
echo "bench rc=$? $(date -u +%T)"
python3 tools/concurrent_check.py http://127.0.0.1:8000 > logs/concurrent_check.log 2>&1; echo "concurrent rc=$?"
python3 tools/sampling_check.py http://127.0.0.1:8000 > logs/sampling_check.log 2>&1; echo "sampling rc=$?"
for n in 300000 500000 1000000; do python3 tools/long_check.py http://127.0.0.1:8000 $n >> logs/long_check_big.log 2>&1; echo "needle $n rc=$? $(date -u +%T)"; done
echo NUMBERS_DONE
