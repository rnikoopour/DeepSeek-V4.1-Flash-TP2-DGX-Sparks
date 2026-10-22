#!/bin/bash
# install_engine.sh (rank 0, after run_build.sh): the built binaries and the input tools into ENGINE_DIR on both nodes.
set -e
source ${DSV41_CONFIG:-${TP2_BASE:-/var/tmp/tf1017-tp2}/config.tp2.env}
read -ra N <<< "$NODES"
E=$BASE/engine
mkdir -p $ENGINE_DIR
cp $E/zig-out/bin/* $E/zig-out/native/bin/tensorfold-native $E/tools/dsv41-zig/inputs/node_inputs.sh \
  $E/tools/dsv41-zig/inputs/make_token_map.py $ENGINE_DIR/
ssh -n -o BatchMode=yes "${N[1]}" "mkdir -p $ENGINE_DIR"
rsync -a --delete $ENGINE_DIR/ "${N[1]}:$ENGINE_DIR/"
ls -la $ENGINE_DIR
