#!/bin/bash
# stop_lane_1017x.sh (root on Reddie): stop the serving lane (vllm_dsv41) on all four nodes, head first, saving the logs under
# /var/tmp/prelaunch-logs/<time>-before-1017x-test. Same stop sequence as the first half of /root/launch.sh. Does NOT launch anything.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
P=/var/tmp/prelaunch-logs/$(date -u +%Y%m%dT%H%M%SZ)-before-1017x-test; mkdir -p $P
if docker ps -a --format '{{.Names}}' | grep -q '^vllm_dsv41$'; then
  docker logs -t vllm_dsv41 > $P/head.log 2>&1; docker rm -f vllm_dsv41 >/dev/null && echo "stopped head (log saved $P/head.log)"
fi
for p in spark4:tonyspark4@192.168.192.4 asusi:tonyspark3@192.168.192.3 bluey:tonyspark1@192.168.192.1; do
  h=${p#*:}; k=${p%%:*}
  if $J $h "docker ps -a --format '{{.Names}}' | grep -q '^vllm_dsv41\$'"; then
    $J $h "docker logs -t vllm_dsv41 2>&1" > $P/worker-$k.log 2>&1; $J $h "docker rm -f vllm_dsv41 >/dev/null" && echo "stopped $k"
  fi
done
chmod 644 $P/*.log 2>/dev/null
date -u +"lane stopped %FT%TZ"
