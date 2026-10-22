#!/bin/bash
# start_tp2.sh (rank 0): checks the pair, then starts the lane on both nodes (serve_tp2.sh).
export DSV41_CONFIG=${DSV41_CONFIG:-/var/tmp/tf1017-tp2/config.tp2.env}
B=$(cd "$(dirname "$0")" && pwd)
$B/precheck_tp2.sh || { echo "precheck failed, not starting"; exit 1; }
$B/serve_tp2.sh start
