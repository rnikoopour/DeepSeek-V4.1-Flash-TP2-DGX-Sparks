#!/bin/bash
# run_build.sh (host): the build container. No GPU is attached.
docker rm -f tf1017tp2-build >/dev/null 2>&1
docker run --rm --name tf1017tp2-build --network host --user "$(id -u):$(id -g)" -v /var/tmp/tf1017-tp2:/work nvcr.io/nvidia/pytorch:26.07-py3 bash /work/build.sh
echo "CONTAINER rc=$?"
