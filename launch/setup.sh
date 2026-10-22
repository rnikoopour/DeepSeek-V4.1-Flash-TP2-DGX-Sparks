#!/bin/bash
# setup.sh (rank 0): Zig 0.17.0 and the engine, patched, under TP2_BASE (default /var/tmp/tf1017-tp2)/engine. Run from this repo's launch/.
set -e
here=$(cd "$(dirname "$0")" && pwd)
B=${TP2_BASE:-/var/tmp/tf1017-tp2}   # every file of the lane lives here (TP2_BASE)
mkdir -p $B && cd $B
[ -d zig-aarch64-linux-0.17.0 ] || { curl -sSL -o zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  zig.tar.xz" | sha256sum -c - && tar xf zig.tar.xz && rm zig.tar.xz; }
[ -d engine ] || git clone -q -b deepseek-v41-zig https://github.com/bertholomus/TensorFold.git engine
cd engine
git checkout -q 3ff3b2a
git checkout -q -B tp2-lane
git -c user.name=tp2-lane -c user.email=tp2-lane@localhost am -q "$here"/../patches/*.patch
git log --oneline -8
cp "$here"/build.sh "$here"/run_build.sh "$here"/node_run.sh "$here"/serve_tp2.sh "$here"/start_tp2.sh "$here"/stop_tp2.sh \
  "$here"/precheck_tp2.sh "$here"/install_engine.sh "$here"/inputs_tp2.sh $B/
[ -f $B/config.tp2.env ] || sed "s|^BASE=.*|BASE=$B|" "$here"/config.tp2.env > $B/config.tp2.env
echo "SETUP DONE: edit $B/config.tp2.env, then $B/run_build.sh"
