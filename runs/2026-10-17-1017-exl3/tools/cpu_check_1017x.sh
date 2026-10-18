#!/usr/bin/env bash
# cpu_check_1017x.sh [extra python args] (run ON a Spark): byte-compile + import the patched modules and run the name-resolution
# test in a throwaway CPU-only container (no --gpus, ~2 GB RAM) with the dsv41-exl3-1017x patch set mounted per mounts.txt.
# Touches no running container. Pass a quantization_config.json path (under /models/...) to test against the real file.
set -euo pipefail
PD="${PATCH_DIR:-$HOME/patches/dsv41-exl3-1017x}"
SITE=/usr/local/lib/python3.12/dist-packages/vllm
MNT=""
while read -r f rel; do
  [ -z "$f" ] && continue
  case "$rel" in /*) tgt="$rel" ;; *) tgt="$SITE/$rel" ;; esac
  MNT="$MNT -v $PD/$f:$tgt:ro"
done < "$PD/mounts.txt"
EXTRA=""
[ -n "${MODEL_HOST:-}" ] && EXTRA="-v $MODEL_HOST:/models/ckpt:ro -e CUDA_EXL3_MODEL_PATH=/models/ckpt"
docker run --rm --network none --entrypoint python3 $MNT $EXTRA \
  -v "$HOME/kai-exl3-1017:/kx:ro" \
  -e CUDA_EXL3_DEBUG_NAMES=0 vllm-dsv41:exl3b-roce -c "
import importlib, sys
for m in ('cuda_exl3.config','vllm.models.deepseek_v4_1.exl3_dense','vllm.models.deepseek_v4_1.attention','vllm.models.deepseek_v4_1.compressor',
          'vllm.models.deepseek_v4_1.nvidia.model','vllm.models.deepseek_v4_1.nvidia.dspark','vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse',
          'vllm.models.deepseek_v4_1.nvidia.vl_model','vllm.models.deepseek_v4.nvidia.model'):
    try:
        importlib.import_module(m); print('import ok  ', m)
    except Exception as e:
        print('IMPORT FAIL', m, type(e).__name__, str(e)[:300]); sys.exit(2)
sys.argv = ['t'] + sys.argv[1:]
exec(open('/kx/test_exl3_names.py').read())
" "$@" 2>&1 | grep -v "^WARNING\|^INFO\|Triton\|cpp_extension\|vllm._C"
