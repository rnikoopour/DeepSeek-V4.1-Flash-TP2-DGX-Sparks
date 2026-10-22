#!/bin/bash
# inputs_tp2.sh (rank 0, the node holding the pack): both ranks' rank caches (WORLD=2) and the token map, CPU only, low
# priority: rank 0's into CACHE_DIR, rank 1's built here, sent to rank 1 and removed. PACK: the downloaded 2.9 bpw pack.
set -e
source ${DSV41_CONFIG:-${TP2_BASE:-/var/tmp/tf1017-tp2}/config.tp2.env}
read -ra N <<< "$NODES"
PACK=${PACK:?set PACK to the downloaded DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw directory}
export IMAGE
echo "== rank 0 $(date -u +%T)"
nice -n 19 ionice -c 3 $ENGINE_DIR/node_inputs.sh $PACK $CACHE_DIR 0 2
echo "== rank 1 $(date -u +%T)"
nice -n 19 ionice -c 3 $ENGINE_DIR/node_inputs.sh $PACK $BASE/cache-r1 1 2
ssh -n -o BatchMode=yes "${N[1]}" "mkdir -p $CACHE_DIR"
rsync -a $BASE/cache-r1/ "${N[1]}:$CACHE_DIR/"
rm -rf $BASE/cache-r1   # rank 1's weight file now lives on rank 1
echo INPUTS_DONE
