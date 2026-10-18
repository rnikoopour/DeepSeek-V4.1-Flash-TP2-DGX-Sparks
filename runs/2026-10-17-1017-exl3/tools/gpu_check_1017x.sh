#!/usr/bin/env bash
# gpu_check_1017x.sh [script.py in ~/kai-exl3-1017] (run ON Reddie, only while the serving lane is stopped):
# the single-GPU loader unit test against the pure-torch oracle, in the vllm-dsv41:exl3b-roce image with the dsv41-exl3-1017x patch
# set mounted per mounts.txt, the EXL3 checkpoint at /models/ckpt and ~/kai-exl3-1017 at /kx. Refuses to run next to a live vllm_dsv41.
set -euo pipefail
if docker ps --format '{{.Names}}' | grep -q '^vllm_dsv41$'; then echo "vllm_dsv41 is running: stop the lane first" >&2; exit 2; fi
PD="${PATCH_DIR:-$HOME/patches/dsv41-exl3-1017x}"
SITE=/usr/local/lib/python3.12/dist-packages/vllm
MNT=""
while read -r f rel; do
  [ -z "$f" ] && continue
  case "$rel" in /*) tgt="$rel" ;; *) tgt="$SITE/$rel" ;; esac
  MNT="$MNT -v $PD/$f:$tgt:ro"
done < "$PD/mounts.txt"
S="${1:-gpu_unit_test_1017x.py}"
docker run --rm --gpus all --network host --ipc host --shm-size 8g --entrypoint python3 $MNT \
  -v /var/tmp/models/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw:/models/ckpt:ro -v "$HOME/kai-exl3-1017:/kx:ro" \
  -e CUDA_EXL3_MODEL_PATH=/models/ckpt -e PYTHONUNBUFFERED=1 -e VLLM_USE_RUST_FRONTEND=0 \
  vllm-dsv41:exl3b-roce "/kx/$S" 2>&1 | grep -v "^WARNING.*vllm._C\|cpp_extension"
