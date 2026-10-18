#!/bin/bash
# prelaunch-1017x.sh (root on Reddie) = prelaunch-1017.sh for the full-EXL3 DeepSeek-V4.1-Flash-1017 lane (LibertAIDAI EXL3 3.5 bpw):
#   checks the EXL3 model dir (shard 44 + quantization_config.json) and the base 1017 checkpoint (Engram tables, shard 47) instead of the native shard 48.
#   BOOT=<N> (go script /root/final/boot$N-go.sh -> Asusi /tmp), PATCH_SET (default dsv41-boot10), IMG (default overlay5).
# Fails (launch.sh then does NOT launch) when: a GLM rank (glm53kd/glm53k/glm53rs) is still running on any node,
#   a worker is unreachable / lacks the 1017 NFS model, a worker lacks the node-local 1017 Engram copy
#   (ALLOW_NFS_ENGRAM=1 to launch anyway), dsv41-tp4-1017x.sh or the patch set's engram.py differs from Reddie's,
#   the image is missing on a node, or a GPU burns under 50 TFLOPS.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
N=${BOOT:?BOOT=<boot label>}
GO=/root/final/boot$N-go.sh
# patch set + image default to the LAST PATCH_NAME= / IMAGE= the go script sets (go scripts stack overrides)
PS=${PATCH_SET:-$(grep -oE "PATCH_NAME=[^ ]+" $GO 2>/dev/null | tail -1 | cut -d= -f2)}; PS=${PS:-dsv41-exl3-1017x}
IMG=${IMG:-$(grep -oE "IMAGE=[^ ]+" $GO 2>/dev/null | tail -1 | cut -d= -f2)}; IMG=${IMG:-vllm-dsv41:overlay5}
M=DeepSeek-V4.1-Flash-1017
MX=DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw
WORKERS="SPARK4:tonyspark4@192.168.192.4 ASUSI:tonyspark3@192.168.192.3 BLUEY:tonyspark1@192.168.192.1"
A=tonyspark3@192.168.192.3
fail=0
EXP_TP4=$(md5sum < /home/tonyspark2/dsv41-tp4-1017x.sh | cut -c1-8)
EXP_ENG=$(cd /home/tonyspark2/patches/$PS && md5sum *.py mounts.txt | md5sum | cut -c1-8)
echo "=== prelaunch-1017x $(date -u +%T): boot $N, patch set $PS, image $IMG, expect dsv41-tp4-1017x.sh $EXP_TP4, $PS/engram.py $EXP_ENG"
g=$(docker ps --format '{{.Names}}' | grep -E '^glm53' | tr '\n' ' ')
[ -z "$g" ] || { echo "  FAIL REDDIE: GLM still running: $g"; fail=1; }
docker image inspect $IMG >/dev/null 2>&1 || { echo "  FAIL REDDIE: image $IMG missing"; fail=1; }
test -f /var/tmp/models/$MX/model-00044-of-00044.safetensors && test -f /var/tmp/models/$MX/quantization_config.json || { echo "  FAIL REDDIE: model $MX incomplete"; fail=1; }
test -f /var/tmp/models/$M/model-00047-of-00048.safetensors || { echo "  FAIL REDDIE: engram base $M incomplete"; fail=1; }
for p in $WORKERS; do
  n=${p%%:*}; h=${p#*:}
  out=$($J -n $h "
    timeout 10 test -f /mnt/reddie-models/$MX/config.json || { sudo -n umount -l /mnt/reddie-models 2>/dev/null; sudo -n mount -t nfs -o ro,vers=3 192.168.192.2:/var/tmp/models /mnt/reddie-models && echo remounted; }
    timeout 10 test -f /mnt/reddie-models/$MX/model-00044-of-00044.safetensors && timeout 10 test -f /mnt/reddie-models/$M/model-00047-of-00048.safetensors && echo mount=ok || echo mount=FAIL
    grep -q '\"source\": \"/mnt/reddie-models/$M\"' /var/tmp/engram-local/$M/engram-local.json 2>/dev/null && echo engram_local=ok || echo engram_local=MISSING
    echo tp4=\$(md5sum < ~/dsv41-tp4-1017x.sh | cut -c1-8) eng=\$(cd ~/patches/$PS 2>/dev/null && md5sum *.py mounts.txt | md5sum | cut -c1-8)
    docker image inspect $IMG >/dev/null 2>&1 && echo image=ok || echo image=MISSING
    echo glm=\$(docker ps --format '{{.Names}}' | grep -E '^glm53' | tr '\n' ,)
    echo up_since=\$(uptime -s | tr ' ' T)" 2>&1 | tr '\n' ' ')
  echo "$n: ${out:-UNREACHABLE}"
  case "$out" in *mount=ok*) ;; *) echo "  FAIL $n: unreachable or model mount missing"; fail=1;; esac
  case "$out" in *engram_local=ok*) ;; *)
    if [ "${ALLOW_NFS_ENGRAM:-0}" = 1 ]; then echo "  WARN $n: no node-local 1017 Engram copy, this rank reads NFS";
    else echo "  FAIL $n: no node-local 1017 Engram copy (tools/engram_local.py, or ALLOW_NFS_ENGRAM=1)"; fail=1; fi;; esac
  echo "$out" | grep -q "tp4=$EXP_TP4 eng=$EXP_ENG" || { echo "  FAIL $n: staged files differ from Reddie"; fail=1; }
  case "$out" in *image=ok*) ;; *) echo "  FAIL $n: image $IMG missing"; fail=1;; esac
  case "$out" in *"glm= "*|*"glm=") ;; *glm=glm53*) echo "  FAIL $n: GLM still running"; fail=1;; esac
done
if [ -f /root/final/boot$N-go.sh ]; then
  $J $A "cat > /tmp/boot$N-go.sh" < /root/final/boot$N-go.sh
  echo "asusi /tmp/boot$N-go.sh $($J -n $A "md5sum < /tmp/boot$N-go.sh | cut -c1-8") (from /root/final)"
else
  echo "  FAIL: /root/final/boot$N-go.sh missing"; fail=1
fi
gib() { awk '/^MemAvailable:/{print int($2/1048576)}'; }
for i in $(seq 1 24); do
  ok=1; [ "$(gib < /proc/meminfo)" -ge 90 ] || ok=0
  for p in $WORKERS; do w=$($J -n ${p#*:} 'cat /proc/meminfo' | gib); [ "${w:-0}" -ge 90 ] || ok=0; done
  [ $ok = 1 ] && break; sleep 5
done
cat > /tmp/burn-quick.sh <<'B'
( docker run --rm --gpus all --network none --entrypoint python3 vllm-dsv41:overlay5 -c "
import torch, time
a = torch.randn(4096, 4096, dtype=torch.float16, device='cuda'); b = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
for _ in range(10): c = a @ b
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < 10:
    c = a @ b; n += 1
torch.cuda.synchronize(); print(f'TFLOPS {2*4096**3*n/(time.time()-t0)/1e12:.1f}')
" 2>&1 | grep TFLOPS ) &
sleep 8; s=$(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader | tr '\n' ' '); wait
echo "under load: $s"
B
echo "--- 10 s fp16 burn (healthy about 75-90 TFLOPS, 2.2-2.4 GHz, 80 W+; latched about 700-950 MHz)"
( echo "REDDIE $(bash /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-REDDIE.txt ) &
for p in $WORKERS; do ( echo "${p%%:*} $($J ${p#*:} 'bash -s' < /tmp/burn-quick.sh 2>&1 | tr '\n' ' ')" > /tmp/burn-${p%%:*}.txt ) & done
wait
for n in REDDIE SPARK4 ASUSI BLUEY; do
  l=$(cat /tmp/burn-$n.txt); echo "$l"
  t=$(echo "$l" | grep -oE 'TFLOPS [0-9.]+' | awk '{print int($2)}')
  [ "${t:-0}" -ge 50 ] || { echo "  FAIL $n: GPU burn ${t:-none} TFLOPS (latched or no GPU): power-cycle with the adapter unplugged 30-60 s"; fail=1; }
done
echo "=== prelaunch-1017x done $(date -u +%T): $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
