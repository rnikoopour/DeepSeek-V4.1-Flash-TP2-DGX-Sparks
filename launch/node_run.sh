#!/bin/bash
# Node side, fed over ssh by serve_tp2.sh (from bertholomus's TP4 recipe scripts/node_run.sh, Apache-2.0): `NAME CMD` runs CMD in a
# detached container NAME with the GPUs, host network, IPC and RDMA devices, the engine, kit, weights and cache mounted, and every TF_*,
# NCCL_* and TENSORFOLD_* variable of the caller passed in. Output goes to $LOG_DIR/NAME.log.
set -u
NAME=${1:?container name}; CMD=${2:?command}
: "${IMAGE:?}" "${ENGINE_DIR:?}" "${KIT_DIR:?}" "${MODEL_DIR:?}" "${ENGRAM_DIR:?}" "${CACHE_DIR:?}" "${LOG_DIR:?}"
mkdir -p "$CACHE_DIR" "$LOG_DIR"
docker rm -f "$NAME" >/dev/null 2>&1
envs=()
while IFS= read -r v; do envs+=(-e "$v"); done < <(env | grep -oE '^(TF_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|TENSORFOLD_[A-Z0-9_]+)=' | tr -d =)
exec docker run -d --rm --name "$NAME" --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK $( [ -d /dev/infiniband ] && echo --device /dev/infiniband ) \
  -v "$ENGINE_DIR:/engine:ro" -v "$KIT_DIR:/kit:ro" -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" \
  -v "$CACHE_DIR:/cache" -v "$LOG_DIR:/logs" "${envs[@]}" "$IMAGE" bash -c "( $CMD ) 2>&1 | tee /logs/$NAME.log"
