#!/bin/bash
# setup.sh (rank 0): Zig 0.17.0 and the engine, patched, under TP2_BASE (default /var/tmp/tf1017-tp2)/engine. Run from this repo's launch/.
# The engine pin is branch head 2aa3c47 (2026-10-23); patches/ holds 0001-0006 (0004 re-ports the shortest-waiting-first
# prompt scheduling that upstream's a27d2f00 did not take). To go back to the 3ff3b2a pin with patches 0004-0007, see ROLLBACK.md.
set -e
here=$(cd "$(dirname "$0")" && pwd)
B=${TP2_BASE:-/var/tmp/tf1017-tp2}   # every file of the lane lives here (TP2_BASE)
mkdir -p $B && cd $B
[ -d zig-aarch64-linux-0.17.0 ] || { curl -sSL -o zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  zig.tar.xz" | sha256sum -c - && tar xf zig.tar.xz && rm zig.tar.xz; }
[ -d engine ] || git clone -q -b deepseek-v41-zig https://github.com/bertholomus/TensorFold.git engine
cd engine
git checkout -q 2aa3c47d5f4b0b57296e69896aedfd3eaab18df0   # branch head 2026-10-23 (was 3ff3b2a: patches 0004-0007 are upstream as 3de79d83, a27d2f00)
git checkout -q -B tp2-lane
git -c user.name=tp2-lane -c user.email=tp2-lane@localhost am -q "$here"/../patches/*.patch
git log --oneline -8
cp "$here"/build.sh "$here"/run_build.sh "$here"/node_run.sh "$here"/serve_tp2.sh "$here"/start_tp2.sh "$here"/stop_tp2.sh \
  "$here"/precheck_tp2.sh "$here"/install_engine.sh "$here"/inputs_tp2.sh $B/
# config.tp2.env always follows this repo; a copy edited in place is kept as .bak (move those edits to site.env)
sed "s|^BASE=.*|BASE=$B|" "$here"/config.tp2.env > $B/config.tp2.env.new
if [ -f $B/config.tp2.env ] && ! cmp -s $B/config.tp2.env $B/config.tp2.env.new; then cp $B/config.tp2.env $B/config.tp2.env.bak; fi
mv $B/config.tp2.env.new $B/config.tp2.env
[ -f $B/site.env ] || cat > $B/site.env <<'SITE'
# This machine's lane settings, sourced at the end of config.tp2.env (see its comments). setup.sh never overwrites it.
NODES="<user>@<rank0-ip> <user>@<rank1-ip>" # rank 0 first: addresses on the direct QSFP link between the two Sparks
MASTER_IP="<rank0-ip>"                     # rank 0's address on that link
SITE
echo "SETUP DONE: edit $B/site.env, then $B/run_build.sh"
