#!/bin/bash
# stop_tp4.sh (on Reddie): PHASE B start. Removes the tp4-lane container on the four nodes (serve.sh stop with the live fleet config).
# It changes no file: restore_tp4.sh brings the same lane back with start_fleet_rail2_p4_ablit.sh.
export DSV41_CONFIG=/var/tmp/tf1017/config.fleet-8000-rail2-p4-ablit.env
date -u
/var/tmp/tf1017/recipe/scripts/serve.sh stop
sleep 5
for t in tonyspark2@192.168.192.2 tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  ssh -n -o BatchMode=yes $t 'echo "$(hostname): containers: $(docker ps --format "{{.Names}}" | tr "\n" " ") gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"'
done
