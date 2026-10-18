#!/bin/bash
echo "$(hostname): avail $(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo) GiB; containers: $(docker ps --format '{{.Names}}' | tr '\n' ' '); caches: $(ls /var/tmp/tf1017/cache/rank-cache | tr '\n' ' '); gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
