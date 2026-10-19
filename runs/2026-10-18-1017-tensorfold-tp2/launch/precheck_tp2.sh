#!/bin/bash
# precheck_tp2.sh (on Reddie): is the pair ready for a TP2 start? Exit 1 on any failed check. Starts nothing.
B=/var/tmp/tf1017-tp2; rc=0
bad() { echo "FAIL: $*"; rc=1; }
# rail 2 runtime address on Reddie (Bluey carries 192.168.193.1 permanently)
bash $B/rail2.sh add 2 >/dev/null 2>&1
for t in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1; do
  out=$(ssh -n -o BatchMode=yes $t "
    echo \"avail \$(awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo)\"
    echo \"containers \$(docker ps --format '{{.Names}}' | tr '\n' ' ')\"
    echo \"gpuprocs \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)\"
    echo \"cache \$(ls $B/cache/rank-cache 2>/dev/null | tr '\n' ' ')\"
    echo \"token \$(ls $B/cache/dsv41_token_map.json 2>/dev/null)\"
    echo \"kit \$(ls $B/kit/MANIFEST $B/kit/vision/gate_bias_vl.safetensors $B/kit/rope-plain-cos.f32 2>/dev/null | wc -l)\"
    echo \"model \$(ls $B/model/config.json $B/model/tokenizer.json $B/model/chat_template.jinja $B/model/model.safetensors.index.json 2>/dev/null | wc -l)\"
    echo \"engine \$(ls $B/engine-bin/tensorfold-native $B/engine-bin/tf-dsv41-lanes 2>/dev/null | wc -l)\"
    echo \"engram \$(ls /var/tmp/tf1017/engram/model-00047-of-00048.safetensors /var/tmp/tf1017/engram/model-00048-of-00048.safetensors 2>/dev/null | wc -l)\"
    echo \"gid3 \$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3 2>/dev/null)\"
    ip -4 -o addr show dev enP2p1s0f0np0 | awk '{print \"rail2\", \$4}'")
  echo "== $t"; echo "$out"
  echo "$out" | grep -q '^containers .*tp4-lane' && bad "$t still runs tp4-lane (stop TP4 first)"
  echo "$out" | grep -q '^containers .*tp2-lane' && bad "$t already runs tp2-lane"
  [ "$(echo "$out" | awk '/^gpuprocs/ {print $2}')" = 0 ] || bad "$t has GPU processes"
  echo "$out" | grep -q '^cache rank[01]of2-' || bad "$t has no rank cache"
  [ "$(echo "$out" | awk '/^kit/ {print $2}')" = 3 ] || bad "$t kit incomplete"
  [ "$(echo "$out" | awk '/^model/ {print $2}')" = 4 ] || bad "$t model dir incomplete"
  [ "$(echo "$out" | awk '/^engine/ {print $2}')" = 2 ] || bad "$t engine binaries missing"
  [ "$(echo "$out" | awk '/^engram/ {print $2}')" = 2 ] || bad "$t engram shards missing"
  a=$(echo "$out" | awk '/^avail/ {print $2}'); awk -v a="$a" 'BEGIN{exit !(a+0 >= 108)}' || bad "$t MemAvailable $a GiB < 108 (the lane needs about 99 GiB of weights and the pool)"
done
ssh -n -o BatchMode=yes tonyspark1@192.168.192.1 'ping -c 1 -W 1 -I enP2p1s0f0np0 192.168.193.2 >/dev/null 2>&1 && echo "rail2 ping Bluey->Reddie ok" || echo "rail2 ping FAILED"' | tee /dev/stderr | grep -q ok || bad "rail 2 not up"
[ $rc = 0 ] && echo "PRECHECK OK" || echo "PRECHECK FAILED"
exit $rc
