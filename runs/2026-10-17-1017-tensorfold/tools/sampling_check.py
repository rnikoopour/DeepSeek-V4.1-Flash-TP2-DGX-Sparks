#!/usr/bin/env python3
"""sampling_check.py BASE: the sampling behaviour the fleet relies on (patch 0004 and the server defaults).
BASE like http://<head-node>:8000"""
import json, sys, urllib.request, urllib.error
base = sys.argv[1].rstrip("/"); fails = 0
PROMPT = "Write a catchy 8-word tagline for a sneaker store."
def ask(extra, prompt=PROMPT, n=40):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": n}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:100].decode("utf-8", "ignore")
def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
outs = [ask({})[1] for _ in range(6)]
check("6 identical requests, no seed, no temperature: outputs differ", len(set(outs)) > 1, "distinct %d of 6" % len(set(outs)))
outs = [ask({"temperature": 0.8})[1] for _ in range(6)]
check("6 identical requests, temperature 0.8, no seed: outputs differ", len(set(outs)) > 1, "distinct %d of 6" % len(set(outs)))
a, b = ask({"seed": 7})[1], ask({"seed": 7})[1]
check("seed=7 twice: identical", a == b, repr(a[:50]))
c = ask({"seed": 8})[1]; d = ask({"seed": 9})[1]; e = ask({"seed": 10})[1]
check("different seeds differ", len({a, c, d, e}) > 1, "distinct %d of 4" % len({a, c, d, e}))
g = [ask({"temperature": 0})[1] for _ in range(4)]
check("temperature 0: deterministic", len(set(g)) == 1, repr(g[0][:50]))
for k in (-1, 0, 100, 56, 1, 20):
    code, txt = ask({"temperature": 0.9, "top_k": k}, n=20)
    check("top_k %d -> 200" % k, code == 200, "" if code == 200 else "%d %s" % (code, txt))
code, txt = ask({"temperature": 0.9, "top_k": None}, n=20); check("top_k null -> 200", code == 200, "" if code == 200 else txt)
code, txt = ask({"temperature": 0.9, "top_p": 0.9, "min_p": 0.05}, n=20); check("top_p + min_p, no top_k -> 200", code == 200, "" if code == 200 else txt)
print("RESULT", "all pass" if fails == 0 else "%d failed" % fails)
