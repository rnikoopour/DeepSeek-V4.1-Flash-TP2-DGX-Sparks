#!/bin/bash
# start_fleet_rail2.sh (on Reddie): the fleet endpoint on both RoCE rails. The rail-2 addresses are runtime-only (a reboot brings the
# link-local ones back), so every start re-adds them first; rail2_all.sh add is idempotent.
export DSV41_CONFIG=/var/tmp/tf1017/config.fleet-8000-rail2.env
date -u
/var/tmp/tf1017/rail2_all.sh add
/var/tmp/tf1017/recipe/scripts/serve.sh start
echo "START rc=$?"
date -u
