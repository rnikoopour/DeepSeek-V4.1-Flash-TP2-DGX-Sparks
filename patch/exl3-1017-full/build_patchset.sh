#!/usr/bin/env bash
# build_patchset.sh: assemble the patch set for the full-EXL3 DeepSeek-V4.1-Flash-1017 lane on one Spark.
#   BASE  an existing EXL3 patch set (default ~/patches/dsv41-exl3-sr4: the EXL3 TP4 lane with the RoCE comm files, the
#         indexer TP-split, the draft shard filter and the cuda-exl3 overlays) -- supplies every file this folder does not change
#   OUT   the new set (default ~/patches/dsv41-exl3-1017x)
# Files in this folder replace or extend the base set; mounts.txt here is the complete manifest.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-$HOME/patches/dsv41-exl3-sr4}"
OUT="${OUT:-$HOME/patches/dsv41-exl3-1017x}"
test -d "$BASE" || { echo "BASE $BASE missing" >&2; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"
cp -a "$BASE"/. "$OUT"/
for f in "$HERE"/*.py "$HERE"/mounts.txt; do cp -p "$f" "$OUT/"; done
# every manifest entry must exist
while read -r f rel; do
  [ -z "$f" ] && continue
  test -f "$OUT/$f" || { echo "MISSING $f" >&2; exit 3; }
done < "$OUT/mounts.txt"
( cd "$OUT" && md5sum *.py mounts.txt | cut -c1-8,33- | tr '\n' ' ' ); echo
echo "patch set $OUT ready ($(wc -l < "$OUT/mounts.txt") mounts)"
