#!/bin/bash
# precheck_tp2.sh (rank 0): is the pair ready for a start? Exit 1 on any failed check. Starts nothing.
source ${DSV41_CONFIG:-${TP2_BASE:-/var/tmp/tf1017-tp2}/config.tp2.env}
read -ra N <<< "$NODES"
hca=${RDMA_HCA%%,*}; rc=0
bad() { echo "FAIL: $*"; rc=1; }
for t in "${N[@]}"; do
  out=$(ssh -n -o BatchMode=yes $t "
    echo \"avail \$(awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo)\"
    echo \"containers \$(docker ps --format '{{.Names}}' | tr '\n' ' ')\"
    echo \"gpuprocs \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)\"
    echo \"cache \$(ls $CACHE_DIR/rank-cache 2>/dev/null | tr '\n' ' ')\"
    echo \"kit \$(ls $KIT_DIR/MANIFEST $KIT_DIR/vision/gate_bias_vl.safetensors $KIT_DIR/rope-plain-cos.f32 2>/dev/null | wc -l)\"
    echo \"model \$(ls $MODEL_DIR/config.json $MODEL_DIR/tokenizer.json $MODEL_DIR/chat_template.jinja $MODEL_DIR/model.safetensors.index.json 2>/dev/null | wc -l)\"
    echo \"engine \$(ls $ENGINE_DIR/tensorfold-native $ENGINE_DIR/tf-dsv41-lanes 2>/dev/null | wc -l)\"
    echo \"engram \$(ls $ENGRAM_DIR/model-00047-of-00048.safetensors $ENGRAM_DIR/model-00048-of-00048.safetensors 2>/dev/null | wc -l)\"
    echo \"gid \$(cat /sys/class/infiniband/$hca/ports/1/gid_attrs/types/\${NCCL_IB_GID_INDEX:-3} 2>/dev/null)\"")
  echo "== $t"; echo "$out"
  # NODES must name an account this user (the service runs as its User=) can reach without a password
  [ -n "$out" ] || { bad "$t not reachable by ssh as $(id -un) (check NODES in $DSV41_CONFIG)"; continue; }
  echo "$out" | grep -q "^containers .*$LANE" && bad "$t already runs $LANE"
  [ "$(echo "$out" | awk '/^gpuprocs/ {print $2}')" = 0 ] || bad "$t has GPU processes"
  echo "$out" | grep -q '^cache rank[01]of2-' || bad "$t has no rank cache"
  [ "$(echo "$out" | awk '/^kit/ {print $2}')" = 3 ] || bad "$t kit incomplete"
  [ "$(echo "$out" | awk '/^model/ {print $2}')" = 4 ] || bad "$t model dir incomplete"
  [ "$(echo "$out" | awk '/^engine/ {print $2}')" = 2 ] || bad "$t engine binaries missing (install_engine.sh)"
  [ "$(echo "$out" | awk '/^engram/ {print $2}')" = 2 ] || bad "$t Engram shards missing"
  a=$(echo "$out" | awk '/^avail/ {print $2}'); awk -v a="$a" 'BEGIN{exit !(a+0 >= 108)}' || bad "$t MemAvailable $a GiB < 108 (the lane needs about 99 GiB of weights and the pool)"
done
[ $rc = 0 ] && echo "PRECHECK OK" || echo "PRECHECK FAILED"
exit $rc
