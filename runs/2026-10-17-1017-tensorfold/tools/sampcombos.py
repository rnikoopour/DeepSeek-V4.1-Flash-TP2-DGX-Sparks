import json, sys, urllib.request, urllib.error
base = sys.argv[1]
def t(label, extra):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Give me one catchy tagline for a sneaker store."}], "max_tokens": 30}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=60).read()); print("OK  ", label, repr(r["choices"][0]["message"]["content"][:50]), r.get("tensorfold", {}).get("sampling"))
    except urllib.error.HTTPError as e:
        print("ERR ", label, e.code, e.read()[:90])
t("no params", {})
t("temperature 0", {"temperature": 0})
t("temperature 0.7", {"temperature": 0.7})
t("temperature 1.0", {"temperature": 1.0})
t("temp 0.7 top_p 0.9", {"temperature": 0.7, "top_p": 0.9})
t("temp 0.7 top_k 20", {"temperature": 0.7, "top_k": 20})
t("temp 0.7 top_k 50", {"temperature": 0.7, "top_k": 50})
t("temp 0.7 top_k 200", {"temperature": 0.7, "top_k": 200})
t("temp 1.0 top_p 0.95 top_k 20", {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
t("temp 0.7 top_p 0.9 top_k 20", {"temperature": 0.7, "top_p": 0.9, "top_k": 20})
t("temp 0.7 min_p 0.05", {"temperature": 0.7, "min_p": 0.05})
