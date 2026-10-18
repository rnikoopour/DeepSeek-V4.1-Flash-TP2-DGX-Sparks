#!/bin/bash
# restore_1017x.sh [N] (root on Reddie): bring up the full-EXL3 DeepSeek-V4.1-Flash-1017 lane with go script /root/final/boot$N-go.sh
#   (default N=1017x). Same flow as restore_1017.sh, with prelaunch-1017x.sh as the gate. Same flow as restore_boot10.sh:
#   launch.sh N (stop vllm_dsv41 on all four, head first, logs saved) -> prelaunch-1017.sh gate (also refuses while GLM runs)
#   -> worker-first fan-out from Asusi -> poll until "Application startup complete" -> postcheck1017.sh.
#   GLM must be stopped first: cd /srv/glm53-knapcio && ENV_FILE=.env.derisked ./start.sh stop
N=${1:-1017x}
L=/var/tmp/restore-$N-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$L") 2>&1
echo "=== restore $N $(date -u +%T) (log $L)"
T0=$(date -u +%s)
BOOT=$N PRELAUNCH=/root/prelaunch-1017x.sh bash /root/launch.sh $N || { echo "launch.sh failed: not launched (see prelaunch output above)"; exit 1; }
for i in $(seq 1 80); do
  c=$(docker inspect vllm_dsv41 --format '{{.Created}}' 2>/dev/null)
  ct=$([ -n "$c" ] && date -u -d "$c" +%s || echo 0)
  [ "$ct" -gt "$T0" ] || { echo "no $N head container on Reddie: stopping"; exit 2; }
  l=$(bash /root/v41poll.sh 2>&1 | tr '\n' ' ')
  [ $((i % 2)) = 1 ] && echo "$(date -u +%T) $l"
  case "$l" in
    *"Application startup complete"*) echo "SERVING $(date -u +%T) ($(( $(date -u +%s) - T0 )) s after kick)"; break;;
    *HEAD-EXITED*|*WORKER-DOWN*) echo "BOOT FAILED $(date -u +%T): $l"; exit 2;;
  esac
  sleep 30
done
case "$l" in *"Application startup complete"*) ;; *) echo "TIMEOUT: not serving after 40 min"; exit 3;; esac
bash /root/postcheck1017.sh boot$N
echo "=== restore $N done $(date -u +%T)"
