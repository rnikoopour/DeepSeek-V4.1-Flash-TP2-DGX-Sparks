#!/bin/bash
# setup.sh (host, any node): Zig 0.17.0, the engine branch and the recipe, under /var/tmp/tf1017-tp2.
set -x
mkdir -p /var/tmp/tf1017-tp2 && cd /var/tmp/tf1017-tp2
[ -d zig-aarch64-linux-0.17.0 ] || { curl -sSL -o zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  zig.tar.xz" | sha256sum -c - && tar xf zig.tar.xz && rm zig.tar.xz; }
[ -d engine ] || git clone -b deepseek-v41-zig https://github.com/bertholomus/TensorFold.git engine
[ -d recipe ] || git clone https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10.git recipe
(cd engine && git checkout -q 3ff3b2a && git checkout -q -b tp2-lane \
  && git cherry-pick 5940208 >/dev/null 2>&1; echo "resolve the cherry-pick as in the README (keep the branch's README and kit tools), then git am patches/0004-*.patch")
echo SETUP DONE
