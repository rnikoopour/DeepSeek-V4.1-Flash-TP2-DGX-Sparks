#!/bin/bash
# start_tp2.sh (on Reddie): PHASE B ONLY, after TP4 is stopped. Re-adds the rail-2 runtime address, checks the pair, starts the TP2 lane.
export DSV41_CONFIG=${DSV41_CONFIG:-/var/tmp/tf1017-tp2/config.tp2.env}
date -u
/var/tmp/tf1017-tp2/precheck_tp2.sh || { echo "precheck failed, not starting"; exit 1; }
/var/tmp/tf1017-tp2/serve_tp2.sh start
echo "START rc=$?"
date -u
