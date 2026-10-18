#!/usr/bin/env bash
# stage_1017x.sh (run from a machine that can ssh to the Sparks, from the repo root): stage the full-EXL3 lane's files on the four
# Sparks WITHOUT touching any running container: the patch set dsv41-exl3-1017x (built from patch/exl3-1017-full on top of
# dsv41-exl3-sr4), the per-rank launcher dsv41-tp4-1017x.sh on all four, the fan-out boot_dsv41_1017x.sh on Asusi.
set -euo pipefail
K="-i $HOME/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
for h in tonyspark2@<tailnet-ip> tonyspark4@<tailnet-ip> tonyspark3@<tailnet-ip> tonyspark1@<tailnet-ip>; do
  echo "== $h"
  tar cf - -C "$REPO/patch" exl3-1017-full | ssh $K "$h" 'rm -rf ~/kai-exl3-1017/exl3-1017-full && mkdir -p ~/kai-exl3-1017 && tar xf - -C ~/kai-exl3-1017 && bash ~/kai-exl3-1017/exl3-1017-full/build_patchset.sh'
  for t in exl3_ref.py gpu_unit_test_1017x.py test_exl3_names.py cpu_check_1017x.sh gpu_check_1017x.sh; do
    cat "$REPO/runs/2026-10-17-1017-exl3/tools/$t" | ssh $K "$h" "cat > ~/kai-exl3-1017/$t && chmod +x ~/kai-exl3-1017/$t"
  done
  cat "$REPO/runs/2026-10-17-1017-exl3/launch/dsv41-tp4-1017x.sh" | ssh $K "$h" 'cat > ~/dsv41-tp4-1017x.sh && chmod +x ~/dsv41-tp4-1017x.sh && md5sum ~/dsv41-tp4-1017x.sh'
done
cat "$REPO/runs/2026-10-17-1017-exl3/launch/boot_dsv41_1017x.sh" | ssh $K tonyspark3@<tailnet-ip> 'cat > ~/boot_dsv41_1017x.sh && md5sum ~/boot_dsv41_1017x.sh'
