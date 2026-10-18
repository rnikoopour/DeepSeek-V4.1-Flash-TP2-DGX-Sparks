"""verify2.py DST SRC MANIFEST: patched regions == blobs, and sampled UNPATCHED regions (outside every patched range,
including the gaps between and around wo_b slices) == stock. Exit 0 only if both hold."""
import json, os, random, sys

dst, src, man = sys.argv[1:4]
rows = json.load(open(man))
d, s = open(dst, "rb"), open(src, "rb")
size = os.path.getsize(dst)
assert size == os.path.getsize(src), "size differs"
blob_ok = changed = 0
for r in rows:
    blob = open(os.path.join(os.path.dirname(man), r["file"]), "rb").read()
    d.seek(r["off"]); got = d.read(r["nbytes"]); s.seek(r["off"]); old = s.read(r["nbytes"])
    blob_ok += got == blob; changed += old != blob
ranges = sorted((r["off"], r["off"] + r["nbytes"]) for r in rows)

def patched(a, b):
    return any(a < hi and b > lo for lo, hi in ranges)

# unpatched samples: right before the first and after the last slice of each layer group, plus 400 random spots
cands = [max(0, ranges[0][0] - 8192), ranges[-1][1]]
for (lo, hi), (lo2, _) in zip(ranges, ranges[1:]):
    if lo2 - hi >= 4096:
        cands.append(hi)  # gap between two patched slices
rnd = random.Random(1017)
cands += [rnd.randrange(0, size - 4096) for _ in range(400)]
checked = unp_bad = 0
for o in cands:
    if patched(o, o + 4096):
        continue
    d.seek(o); s.seek(o); checked += 1
    unp_bad += d.read(4096) != s.read(4096)
print(f"VERIFY2 patched {blob_ok}/{len(rows)} match blob, {changed} differ from stock | unpatched spots {checked} checked, {unp_bad} differ")
sys.exit(0 if blob_ok == len(rows) and unp_bad == 0 else 1)
