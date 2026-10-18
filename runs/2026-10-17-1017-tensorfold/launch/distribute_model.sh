#!/bin/bash
for t in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  rsync -a /var/tmp/tf1017/model $t:/var/tmp/tf1017/ && echo "$t model synced"
done
