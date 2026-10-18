#!/bin/bash
cd /var/tmp/tf1017/kit
./verify_kit.sh /var/tmp/models/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw /var/tmp/models/DeepSeek-V4.1-Flash-1017
echo "VERIFY rc=$?"
