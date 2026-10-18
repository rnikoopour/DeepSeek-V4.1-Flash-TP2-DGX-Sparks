#!/usr/bin/env bash
# collect_results_1017x.sh <label> (run from the repo root on the Mac): copy /var/tmp/boot-results/<label>/ from Reddie (bench json/md,
# report, postcheck, vision-tools, head log) and the newest restore log into runs/2026-10-17-1017-exl3/results/<label>/.
set -euo pipefail
L="${1:?label, e.g. boot1017x}"
K="-i $HOME/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o BatchMode=yes"
D="runs/2026-10-17-1017-exl3/results/$L"
mkdir -p "$D"
ssh $K tonyspark2@<tailnet-ip> "cd /var/tmp/boot-results/$L && tar cf - ." | tar xf - -C "$D"
ssh $K tonyspark2@<tailnet-ip> 'ls -t /var/tmp/restore-*-*.log 2>/dev/null | head -1 | xargs cat' > "$D/restore.log" || true
ls -la "$D"
