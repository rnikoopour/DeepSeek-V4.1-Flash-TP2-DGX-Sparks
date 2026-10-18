#!/bin/bash
set -e
W=/var/tmp/tf1017/tmpl
nice -n 19 $W/parity $W/corpus.json $W/results.json
ls -la $W/results.json
