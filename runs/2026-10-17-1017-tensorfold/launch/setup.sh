#!/bin/bash
set -x
cd /var/tmp/tf1017
[ -d zig-aarch64-linux-0.17.0 ] || { curl -sSL -o zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  zig.tar.xz" | sha256sum -c - && tar xf zig.tar.xz && rm zig.tar.xz; }
[ -d engine ] || git clone -b deepseek-v41-zig-tp4 https://github.com/bertholomus/TensorFold.git engine
[ -d recipe ] || git clone https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10.git recipe
(cd engine && git checkout -q 59402086381a42c74d53053443224eec2790b4c5 && git log --oneline | head -1)
ls
echo SETUP DONE
