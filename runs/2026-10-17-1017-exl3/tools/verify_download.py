"""Verify the LibertAI EXL3 download against the Hugging Face tree listing, without reading tensor data.

usage: verify_download.py <model_dir> <tree.json>
Checks, per file in the tree: exists, size equals the listed size; for *.safetensors also that the file size equals
8 + header length + the largest data offset in its header (a truncated or corrupt shard fails this).
Also writes the full tensor inventory (name -> shard, dtype, shape, nbytes) to <model_dir>/../kai-inventory-<dirname>.json.
"""
import json, os, struct, sys

d, tree = sys.argv[1], sys.argv[2]
t = json.load(open(tree))
bad, total, n = [], 0, 0
inv = {}
for e in t:
    if e["type"] != "file":
        continue
    p = os.path.join(d, e["path"])
    want = e.get("size", 0)
    if not os.path.isfile(p):
        bad.append((e["path"], "missing")); continue
    got = os.path.getsize(p)
    if got != want:
        bad.append((e["path"], f"size {got} != {want}")); continue
    total += got; n += 1
    if e["path"].endswith(".safetensors"):
        with open(p, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hl))
        end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        if 8 + hl + end != got:
            bad.append((e["path"], f"header says {8+hl+end}, file is {got}"))
        for k, v in hdr.items():
            if k != "__metadata__":
                inv[k] = (e["path"], v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0])
out = os.path.join(os.path.dirname(os.path.abspath(d)), f"kai-inventory-{os.path.basename(d.rstrip('/'))}.json")
json.dump(inv, open(out, "w"))
print(f"{n} files, {total/2**30:.2f} GiB verified, {len(inv)} tensors; inventory {out}")
if bad:
    print("PROBLEMS:"); [print(" ", b) for b in bad]; sys.exit(1)
print("DOWNLOAD VERIFIED")
