#!/bin/bash
# inputs_worker.sh RANK: this worker's rank cache and token map from the pack, read over NFS from Reddie.
R=${1:?rank}
export IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
date -u
time /var/tmp/tf1017/engine-bin/node_inputs.sh /mnt/reddie-models/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw /var/tmp/tf1017/cache $R
echo "INPUTS rc=$?"
date -u
ls -la /var/tmp/tf1017/cache /var/tmp/tf1017/cache/rank-cache
