#!/bin/bash
# distribute.sh (on Reddie as tonyspark2): engine-bin, kit and a small model dir (config + tokenizer files, no weights) to the three workers.
for t in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  rsync -a /var/tmp/tf1017/engine-bin /var/tmp/tf1017/kit $t:/var/tmp/tf1017/ && echo "$t synced"
done
