#!/bin/bash
# inputs_tp2.sh (on the node holding the pack): both ranks' rank caches (WORLD=2) and the token map, CPU only, low priority.
export IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
B=/var/tmp/tf1017-tp2
PACK=${PACK:-/var/tmp/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}
date -u
for r in 0 1; do
  echo "== rank $r $(date -u +%T)"
  nice -n 19 ionice -c 3 $B/engine-bin/node_inputs.sh $PACK $B/cache-r$r $r 2
  echo "rank $r rc=$?"
done
date -u
ls -la $B/cache-r0/rank-cache $B/cache-r1/rank-cache $B/cache-r0/dsv41_token_map.json
echo INPUTS_DONE
