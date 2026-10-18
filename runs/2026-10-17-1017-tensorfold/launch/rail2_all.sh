#!/bin/bash
# rail2_all.sh add|del|status   (on Reddie as the node-to-node ssh user): runs rail2.sh on Reddie (N=2) and Asusi (N=3), then pings the
# four rail-2 addresses (192.168.193.1 Bluey, .2 Reddie, .3 Asusi, .4 Spark4) from each of them.
D=$(cd "$(dirname "$0")" && pwd); A=${1:?add|del|status}
bash $D/rail2.sh $A 2
scp -q $D/rail2.sh tonyspark3@192.168.192.3:/tmp/rail2.sh && ssh -o BatchMode=yes tonyspark3@192.168.192.3 "bash /tmp/rail2.sh $A 3"
for t in tonyspark2@192.168.192.2 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1 tonyspark4@192.168.192.4; do
  echo "== ping from $t"; ssh -o BatchMode=yes $t 'for ip in 192.168.193.1 192.168.193.2 192.168.193.3 192.168.193.4; do ping -c 1 -W 1 -I enP2p1s0f0np0 $ip >/dev/null 2>&1 && echo -n "$ip ok  " || echo -n "$ip --  "; done; echo'
done
