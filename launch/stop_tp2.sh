#!/bin/bash
# stop_tp2.sh (rank 0): removes the lane's container on both nodes. Kills nothing else.
export DSV41_CONFIG=${DSV41_CONFIG:-/var/tmp/tf1017-tp2/config.tp2.env}
"$(cd "$(dirname "$0")" && pwd)"/serve_tp2.sh stop
