#!/bin/bash
# stop_tp2.sh (on Reddie): removes the tp2-lane container on both nodes. Kills nothing else.
export DSV41_CONFIG=${DSV41_CONFIG:-/var/tmp/tf1017-tp2/config.tp2.env}
/var/tmp/tf1017-tp2/serve_tp2.sh stop
sleep 3
for t in tonyspark2@192.168.192.2 tonyspark1@192.168.192.1; do ssh -n -o BatchMode=yes $t 'echo "$(hostname): containers: $(docker ps --format "{{.Names}}" | tr "\n" " ") gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"'; done
