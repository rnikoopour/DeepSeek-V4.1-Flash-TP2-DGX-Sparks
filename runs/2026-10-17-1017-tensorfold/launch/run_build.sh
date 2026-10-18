#!/bin/bash
# run_build.sh (host, Reddie): the build container.
docker rm -f tf1017-build >/dev/null 2>&1
docker run --rm --name tf1017-build --network host --user "$(id -u):$(id -g)" -v /var/tmp/tf1017:/work nvcr.io/nvidia/pytorch:26.07-py3 bash /work/build.sh
echo "CONTAINER rc=$?"
