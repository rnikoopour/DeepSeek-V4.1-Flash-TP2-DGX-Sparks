import json, struct, sys, glob, os, collections, re
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
wm = idx["weight_map"]
print("n tensors", len(wm))
files = sorted(set(wm.values()))
hdrs = {}
for f in files:
    with open(os.path.join(d, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdrs[f] = json.loads(fh.read(n))
# group by pattern
pat = collections.defaultdict(lambda: [0, None, set()])
def norm(k):
    k = re.sub(r"\.\d+\.", ".N.", k)
    k = re.sub(r"\.\d+\.", ".N.", k)
    k = re.sub(r"\.\d+$", ".N", k)
    return k
for k, f in wm.items():
    h = hdrs[f][k]
    p = norm(k)
    e = pat[p]
    e[0] += 1
    if e[1] is None: e[1] = (h["dtype"], h["shape"])
    e[2].add((h["dtype"], tuple(h["shape"])))
for p in sorted(pat):
    e = pat[p]
    print(p, e[0], e[1] if len(e[2]) == 1 else ("MULTI", sorted(e[2])[:4], len(e[2])))
