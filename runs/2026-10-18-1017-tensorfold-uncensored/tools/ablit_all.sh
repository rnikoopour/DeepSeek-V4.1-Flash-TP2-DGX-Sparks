#!/bin/bash
# ablit_all.sh (on Reddie): ship the blobs to the three workers and build every node's patched cache copy in parallel.
set -uo pipefail
W=/var/tmp/tf1017-ablit
declare -A H=([1]=tonyspark4@192.168.192.4 [2]=tonyspark3@192.168.192.3 [3]=tonyspark1@192.168.192.1)
for g in 1 2 3; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 ${H[$g]} "mkdir -p $W/overlay $W/blobs" &&
  scp -q $W/ablit_node.sh ${H[$g]}:$W/ && scp -q $W/overlay/apply_rank_blobs.py ${H[$g]}:$W/overlay/ &&
  scp -q -r $W/blobs/r$g ${H[$g]}:$W/blobs/ || { echo "ship to rank $g failed"; exit 1; }
done
echo "shipped $(date -u +%T)"
bash $W/ablit_node.sh 0 > $W/node0.log 2>&1 &
for g in 1 2 3; do ssh -n -o BatchMode=yes ${H[$g]} "bash $W/ablit_node.sh $g" > $W/node$g.log 2>&1 & done
wait
tail -n 3 $W/node*.log
