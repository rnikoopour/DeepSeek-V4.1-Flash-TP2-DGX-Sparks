#!/bin/bash
export DSV41_CONFIG=/var/tmp/tf1017/config.env
date -u
/var/tmp/tf1017/recipe/scripts/serve.sh start
echo "START rc=$?"
date -u
