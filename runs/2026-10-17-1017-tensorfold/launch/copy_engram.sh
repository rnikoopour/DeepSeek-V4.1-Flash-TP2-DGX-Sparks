#!/bin/bash
# copy_engram.sh: this node's local copy of the two Engram table shards (DeepSeek-V4.1-Flash-1017 shards 47 and 48), read over NFS from Reddie.
D=/var/tmp/tf1017/engram; mkdir -p $D
S=/mnt/reddie-models/DeepSeek-V4.1-Flash-1017
echo "start $(date -u +%FT%TZ)"
for n in 47 48; do
  f=model-000$n-of-00048.safetensors
  if [ "$(stat -c %s $D/$f 2>/dev/null)" = "$(stat -c %s $S/$f)" ]; then echo "$f already complete"; continue; fi
  dd if=$S/$f of=$D/$f bs=16M iflag=direct oflag=direct 2>&1 | tail -1
done
cp $S/model.safetensors.index.json $D/
ls -la $D
echo "done $(date -u +%FT%TZ)"
