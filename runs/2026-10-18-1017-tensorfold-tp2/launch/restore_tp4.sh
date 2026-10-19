#!/bin/bash
# restore_tp4.sh (on Reddie): PHASE B end. Stops the TP2 lane on both nodes and starts the TP4 fleet lane exactly as it was
# (start_fleet_rail2_p4_ablit.sh, config.fleet-8000-rail2-p4-ablit.env, CACHE_DIR=/var/tmp/tf1017/cache-ablit).
date -u
/var/tmp/tf1017-tp2/stop_tp2.sh
for t in tonyspark2@192.168.192.2 tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  ssh -n -o BatchMode=yes $t 'docker ps --format "{{.Names}}" | grep -q "^tp2-lane$" && echo "$(hostname): tp2-lane STILL UP" || true'
done
/var/tmp/tf1017/start_fleet_rail2_p4_ablit.sh
echo "RESTORE rc=$?"
date -u
