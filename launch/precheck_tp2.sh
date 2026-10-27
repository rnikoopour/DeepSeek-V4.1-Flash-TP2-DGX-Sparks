#!/bin/bash
# precheck_tp2.sh (rank 0): is the pair ready for a start? Exit 1 on any failed check. Starts nothing.
# Per node: memory headroom, no lane or GPU process already running, exactly one rank cache and it is this node's rank,
# the token map, kit, model files, engine, Engram shards, the RoCE GID type and every RDMA_HCA port up. If the config
# sets EXPECT_CONFIG_SHA / EXPECT_KMAP_SHA / EXPECT_INDEX_SHA, MODEL_DIR's files must also match those sha256s.
CFG=${DSV41_CONFIG:-${TP2_BASE:-/var/tmp/tf1017-tp2}/config.tp2.env}
[ -r "$CFG" ] || { echo "FAIL: no $CFG"; exit 1; }
source "$CFG"
read -ra N <<< "$NODES"
hca=${RDMA_HCA%%,*}; rc=0; rank=0
bad() { echo "FAIL: $*"; rc=1; }
pin() {   # pin NODE-OUTPUT KEY EXPECTED FILE
  [ -z "$3" ] || [ "$(echo "$1" | awk -v k="$2" '$1 == k {print $2}')" = "$3" ] || bad "$t $4 does not match its pinned sha256"
}
for t in "${N[@]}"; do
  out=$(ssh -n -o BatchMode=yes $t "
    echo \"avail \$(awk '/MemAvailable/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo)\"
    echo \"containers \$(docker ps --format '{{.Names}}' | tr '\n' ' ')\"
    echo \"gpuprocs \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)\"
    echo \"cache \$(ls '$CACHE_DIR'/rank-cache/*.bin 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')\"
    echo \"caches \$(ls '$CACHE_DIR'/rank-cache/*.bin 2>/dev/null | wc -l)\"
    echo \"token \$(ls '$CACHE_DIR'/dsv41_token_map.json 2>/dev/null | wc -l)\"
    echo \"kit \$(ls '$KIT_DIR'/MANIFEST '$KIT_DIR'/vision/gate_bias_vl.safetensors '$KIT_DIR'/rope-plain-cos.f32 2>/dev/null | wc -l)\"
    echo \"model \$(ls '$MODEL_DIR'/config.json '$MODEL_DIR'/tokenizer.json '$MODEL_DIR'/chat_template.jinja '$MODEL_DIR'/model.safetensors.index.json 2>/dev/null | wc -l)\"
    echo \"engine \$(ls '$ENGINE_DIR'/tensorfold-native '$ENGINE_DIR'/tf-dsv41-lanes 2>/dev/null | wc -l)\"
    echo \"engram \$(ls '$ENGRAM_DIR'/model-00047-of-00048.safetensors '$ENGRAM_DIR'/model-00048-of-00048.safetensors 2>/dev/null | wc -l)\"
    echo \"gid \$(cat /sys/class/infiniband/$hca/ports/1/gid_attrs/types/\${NCCL_IB_GID_INDEX:-3} 2>/dev/null)\"
    for h in ${RDMA_HCA//,/ }; do echo \"port \$h \$(cut -d' ' -f2 /sys/class/infiniband/\$h/ports/1/state 2>/dev/null || echo missing)\"; done
    echo \"pin-config \$(sha256sum '$MODEL_DIR'/config.json 2>/dev/null | cut -c1-64)\"
    echo \"pin-kmap \$(sha256sum '$MODEL_DIR'/exl3_k_map.json 2>/dev/null | cut -c1-64)\"
    echo \"pin-index \$(sha256sum '$MODEL_DIR'/model.safetensors.index.json 2>/dev/null | cut -c1-64)\"")
  echo "== $t (rank $rank)"; echo "$out"
  # NODES must name an account this user (the service runs as its User=) can reach without a password
  [ -n "$out" ] || { bad "$t not reachable by ssh as $(id -un) (check NODES in $CFG)"; rank=$((rank+1)); continue; }
  echo "$out" | grep -q "^containers .*$LANE" && bad "$t already runs $LANE"
  [ "$(echo "$out" | awk '/^gpuprocs/ {print $2}')" = 0 ] || bad "$t has GPU processes"
  # the engine takes the one cache file in the folder and refuses a folder with several
  n=$(echo "$out" | awk '/^caches/ {print $2}')
  [ "$n" = 1 ] || bad "$t has $n rank caches in $CACHE_DIR (the lane needs exactly one)"
  echo "$out" | grep -q "^cache rank${rank}of2-" || bad "$t has no rank${rank}of2 cache in $CACHE_DIR"
  [ "$(echo "$out" | awk '/^token/ {print $2}')" = 1 ] || bad "$t has no token map in $CACHE_DIR"
  [ "$(echo "$out" | awk '/^kit/ {print $2}')" = 3 ] || bad "$t kit incomplete"
  [ "$(echo "$out" | awk '/^model/ {print $2}')" = 4 ] || bad "$t model dir incomplete ($MODEL_DIR)"
  [ "$(echo "$out" | awk '/^engine/ {print $2}')" = 2 ] || bad "$t engine binaries missing ($ENGINE_DIR; install_engine.sh)"
  [ "$(echo "$out" | awk '/^engram/ {print $2}')" = 2 ] || bad "$t Engram shards missing"
  echo "$out" | grep -q '^gid RoCE v2' || bad "$t GID index ${NCCL_IB_GID_INDEX:-3} on $hca is not RoCE v2"
  echo "$out" | awk '/^port / && $3 != "ACTIVE" {bad=1} END {exit bad}' || bad "$t has an RDMA_HCA port that is not ACTIVE"
  pin "$out" pin-config "$EXPECT_CONFIG_SHA" config.json
  pin "$out" pin-kmap "$EXPECT_KMAP_SHA" exl3_k_map.json
  pin "$out" pin-index "$EXPECT_INDEX_SHA" model.safetensors.index.json
  a=$(echo "$out" | awk '/^avail/ {print $2}'); awk -v a="$a" 'BEGIN{exit !(a+0 >= 108)}' || bad "$t MemAvailable $a GiB < 108 (the lane needs about 99 GiB of weights and the pool)"
  rank=$((rank+1))
done
[ $rc = 0 ] && echo "PRECHECK OK" || echo "PRECHECK FAILED"
exit $rc
