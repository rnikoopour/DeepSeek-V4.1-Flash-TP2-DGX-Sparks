#!/bin/bash
# run_bench.sh LABEL [extra args]: the standard v1 prompt set against the TensorFold lane on :8001 (completions, DeepSeek encoder).
L=$1; shift
O=/var/tmp/tf1017/bench/$L; mkdir -p $O
echo "bench $L start $(date -u +%FT%TZ)" > $O/bench.txt
python3 /var/tmp/tf1017/v41bench_tf.py --base http://127.0.0.1:8001/v1 --model deepseek-v4.1-flash --label $L --out $O "$@" >> $O/bench.txt 2>&1
echo "BENCH DONE rc=$? $(date -u +%FT%TZ)" >> $O/bench.txt
sudo python3 /root/bench_report.py $O/bench-$L.json > $O/report.md 2>&1
