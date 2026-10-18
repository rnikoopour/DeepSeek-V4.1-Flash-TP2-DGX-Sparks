"""compare_bench.py NATIVE.json RUN.json [FRESH_PREFILL.json]: C1-C6 aggregate / per-stream tok/s, C1 per category, cold prefill, side by side."""
import json, statistics as st, sys
CATS = ["coding", "json", "math", "reasoning", "format", "summary", "prose", "narrative"]
a, b = (json.load(open(p)) for p in sys.argv[1:3])
# A full bench run right after a screen run hits the prefix cache at the 8K / 32K prompts (TTFT under 1 s, tens of thousands of
# tok/s). A third file (the screen run on the fresh boot) supplies the uncached cells for those.
fresh = {p["prompt_tokens"]: p for p in json.load(open(sys.argv[3])).get("prefill", [])} if len(sys.argv) > 3 else {}
def cell(d, c, key, cat=None):
    xs = [x[key] for x in d["batches"] if x["c"] == c and (x["category"] == cat if cat else x["category"] in CATS)]
    return st.mean(xs)
print(f"| C | aggregate tok/s {a['label']} | {b['label']} | ratio | per-stream {a['label']} | {b['label']} |")
print("|---|---|---|---|---|---|")
for c in range(1, 7):
    x, y = cell(a, c, "agg_tok_s"), cell(b, c, "agg_tok_s")
    print(f"| C{c} | {x:.1f} | {y:.1f} | {y/x:.2f} | {cell(a,c,'per_stream_tok_s'):.1f} | {cell(b,c,'per_stream_tok_s'):.1f} |")
print()
print(f"| C1 per-stream tok/s | {a['label']} | {b['label']} | ratio |")
print("|---|---|---|---|")
for cat in CATS + ["ceiling_count"]:
    x, y = cell(a, 1, "per_stream_tok_s", cat), cell(b, 1, "per_stream_tok_s", cat)
    print(f"| {cat} | {x:.1f} | {y:.1f} | {y/x:.2f} |")
print()
print("| cold prefill (prompt tokens) | " + a["label"] + " tok/s | " + b["label"] + " tok/s |")
print("|---|---|---|")
pa = {p["prompt_tokens"]: p for p in a.get("prefill", [])}
pb = {p["prompt_tokens"]: p for p in b.get("prefill", [])}
for k, p in list(pb.items()):
    if p["prefill_tok_s"] > 10000 and k in fresh:
        pb[k] = dict(fresh[k], prefill_tok_s=fresh[k]["prefill_tok_s"])
for k in sorted(pa):
    print(f"| {k:,} | {pa[k]['prefill_tok_s']:,.0f} | {pb[k]['prefill_tok_s']:,.0f} |" if k in pb else f"| {k:,} | {pa[k]['prefill_tok_s']:,.0f} | |")
