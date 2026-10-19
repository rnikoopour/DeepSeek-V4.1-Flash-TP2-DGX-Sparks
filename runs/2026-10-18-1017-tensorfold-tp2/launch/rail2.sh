#!/bin/bash
# rail2.sh add|del|status N   (on a node, needs sudo)  N = the last octet of the node's rail-2 address (Reddie 2, Asusi 3).
# Rail 2 is the second RoCE port (enP2p1s0f0np0 / roceP2p1s0f0). Bluey and Spark4 already carry 192.168.193.1 and .4; Reddie and Asusi
# carry only a link-local 169.254.x.x/16 from NetworkManager, so RoCE v2 between mixed subnets cannot connect. This swaps that
# link-local address for 192.168.193.N/24 at RUNTIME only (netplan and the NetworkManager profile are untouched; a reboot brings
# the link-local back, so the boot script runs `rail2.sh add N` again).
DEV=enP2p1s0f0np0; HCA=roceP2p1s0f0
N=${2:?last octet}
case ${1:?add|del|status} in
  add)
    if ip -4 -o addr show dev $DEV | grep -q "192.168.193.$N/"; then echo "already has 192.168.193.$N"; else
      for a in $(ip -4 -o addr show dev $DEV scope link | awk '{print $4}'); do sudo ip addr del "$a" dev $DEV && echo "removed $a"; done
      sudo ip addr add 192.168.193.$N/24 dev $DEV && echo "added 192.168.193.$N/24"
    fi ;;
  del)
    sudo ip addr del 192.168.193.$N/24 dev $DEV && echo "removed 192.168.193.$N/24"
    sudo nmcli connection up netplan-$DEV >/dev/null && echo "link-local profile re-applied" ;;
  status) ;;
esac
ip -4 -o addr show dev $DEV | awk '{print $2, $4}'
for i in 0 1 2 3 4 5; do g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null); t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null)
  [ -n "$g" ] && [ "$g" != "0000:0000:0000:0000:0000:0000:0000:0000" ] && echo "  $HCA gid[$i] $g $t"; done
