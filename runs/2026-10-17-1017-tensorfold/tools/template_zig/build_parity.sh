#!/bin/bash
# Builds TensorFold's template-engine parity program (core/template/parity.zig) from a copy of its sources; CPU only.
set -e
W=/var/tmp/tf1017/tmpl
export ZIG_GLOBAL_CACHE_DIR=$W/zigcache
export ZIG_LOCAL_CACHE_DIR=$W/zigcache-local
rm -rf $W/zigtpl && mkdir -p $W/zigtpl
cp /var/tmp/tf1017/engine/zig/src/core/template/*.zig $W/zigtpl/
cd $W/zigtpl
/var/tmp/tf1017/zig-aarch64-linux-0.17.0/zig version
nice -n 19 /var/tmp/tf1017/zig-aarch64-linux-0.17.0/zig build-exe parity.zig -lc -O ReleaseSafe -femit-bin=$W/parity
ls -la $W/parity
