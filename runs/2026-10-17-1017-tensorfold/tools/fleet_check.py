#!/usr/bin/env python3
"""fleet_check.py BASE: the calls the fleet makes to the model endpoint, against a TensorFold lane on :8000.
BASE like http://<head-node>:8000 . Prints PASS/FAIL/INFO lines; reports how reasoning_effort and
chat_template_kwargs are treated (it does not judge them)."""
import base64, concurrent.futures as cf, json, struct, sys, time, urllib.request, zlib
base = sys.argv[1].rstrip("/"); M = "deepseek-v4.1-flash"
def call(path, body=None, headers=None, timeout=300, raw=False):
    h = {"Content-Type": "application/json"}; h.update(headers or {})
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if raw else json.loads(data)
def out(tag, ok, extra=""): print(("PASS " if ok else "FAIL ") + tag, extra, flush=True)
def info(tag, extra=""): print("INFO " + tag, extra, flush=True)
def chat(extra, msgs=None, timeout=300):
    b = {"model": M, "messages": msgs or [{"role": "user", "content": "How many r letters are in strawberry? Answer briefly."}]}
    b.update(extra); return call("/v1/chat/completions", b, timeout=timeout)
def parts(r):
    m = r["choices"][0]["message"]; rs = m.get("reasoning_content") or m.get("reasoning") or ""
    return m, rs
ids = [m["id"] for m in call("/v1/models")["data"]]
out("models lists both names", "deepseek-v4.1-flash" in ids and "glm-5.3-flash" in ids, str(ids))
h = call("/health"); out("/health", h.get("status") == "ok", json.dumps(h)[:120])
r = chat({"max_tokens": 64, "temperature": 0}); out("chat non-streaming", bool(r["choices"][0]["message"]["content"]), repr(r["choices"][0]["message"]["content"][:60]) + " model=" + r["model"])
r = call("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16, "temperature": 0}); out("alias glm-5.3-flash", bool(r["choices"][0]["message"]["content"]), "reports model=" + r["model"])
# streaming SSE
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"model": M, "messages": [{"role": "user", "content": "Count from 1 to 5."}], "max_tokens": 60, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
n = 0; usage = None; done = False; text = ""; role = False
with urllib.request.urlopen(req, timeout=120) as resp:
    ctype = resp.headers.get("Content-Type")
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": done = True; break
        ev = json.loads(d); n += 1
        if ev.get("usage"): usage = ev["usage"]
        for ch in ev.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("role"): role = True
            text += dl.get("content") or ""
out("chat stream:true SSE", done and n > 2 and bool(text) and usage is not None, "content-type=%s chunks=%d role_delta=%s usage=%s text=%r" % (ctype, n, role, usage and usage.get("completion_tokens"), text[:30]))
# tool call, tool_choice auto, then required/none quick
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Tokyo? Use celsius."}]
r = chat({"messages": msgs, "tools": tools, "tool_choice": "auto", "max_tokens": 200, "temperature": 0})
m, _ = parts(r); tc = m.get("tool_calls") or []
out("tool call, tool_choice auto", len(tc) == 1 and tc[0]["function"]["name"] == "get_weather" and r["choices"][0]["finish_reason"] == "tool_calls", json.dumps(tc)[:150])
# streamed tool call
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"model": M, "messages": msgs, "tools": tools, "tool_choice": "auto", "max_tokens": 200, "stream": True, "temperature": 0}).encode(), headers={"Content-Type": "application/json"})
acc = {}; fin = None
with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
        ev = json.loads(line[5:])
        for ch in ev.get("choices") or []:
            fin = ch.get("finish_reason") or fin
            for t in (ch.get("delta") or {}).get("tool_calls") or []:
                a = acc.setdefault(t.get("index", 0), {"name": "", "args": ""}); f = t.get("function") or {}
                a["name"] += f.get("name") or ""; a["args"] += f.get("arguments") or ""
ok = bool(acc) and acc[0]["name"] == "get_weather" and "tokyo" in acc[0]["args"].lower()
out("tool call streamed", ok, json.dumps(acc)[:150] + " finish=" + str(fin))
# reasoning controls
for label, extra in [("no params (server default)", {}),
                     ("reasoning_effort none", {"reasoning_effort": "none"}),
                     ("reasoning_effort low", {"reasoning_effort": "low"}),
                     ("reasoning_effort medium", {"reasoning_effort": "medium"}),
                     ("reasoning_effort high", {"reasoning_effort": "high"}),
                     ("reasoning_effort max", {"reasoning_effort": "max"}),
                     ("chat_template_kwargs enable_thinking true", {"chat_template_kwargs": {"enable_thinking": True}}),
                     ("chat_template_kwargs thinking true", {"chat_template_kwargs": {"thinking": True}}),
                     ("chat_template_kwargs enable_thinking false", {"chat_template_kwargs": {"enable_thinking": False}}),
                     ("none + enable_thinking true", {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}})]:
    b = {"max_tokens": 1500, "temperature": 0}; b.update(extra)
    try:
        r = chat(b); m, rs = parts(r); u = r["usage"]
        info("reasoning: " + label, "thinking=%s reasoning_chars=%d content=%r completion_tokens=%d reasoning_tokens=%s" % (bool(rs), len(rs), (m.get("content") or "")[:50], u["completion_tokens"], (u.get("completion_tokens_details") or {}).get("reasoning_tokens")))
    except Exception as e:
        info("reasoning: " + label, "ERROR %s %s" % (e, getattr(e, "read", lambda: b"")()[:200]))
# no max_tokens
t0 = time.time(); r = chat({"temperature": 0, "messages": [{"role": "user", "content": "Write a haiku about rain."}]}); out("request with no max_tokens", r["choices"][0]["finish_reason"] == "stop", "%d tokens %.1fs" % (r["usage"]["completion_tokens"], time.time() - t0))
# image
def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
img = "data:image/png;base64," + base64.b64encode(png(448, 448, (20, 40, 220))).decode()
r = chat({"max_tokens": 16, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "text", "text": "What color is this image? One word."}, {"type": "image_url", "image_url": {"url": img}}]}]})
c = r["choices"][0]["message"]["content"]; out("image input", "blue" in c.lower(), repr(c))
# Authorization header variants (the fleet sends Bearer <anything> or none)
for hdr in ({}, {"Authorization": "Bearer sk-anything"}, {"Authorization": "Bearer EMPTY"}):
    try:
        r = call("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 4, "temperature": 0}, headers=hdr)
        out("auth header %s" % (list(hdr.values())[0] if hdr else "none"), bool(r["choices"][0]["message"]["content"]))
    except Exception as e:
        out("auth header %s" % (list(hdr.values())[0] if hdr else "none"), False, str(e))
# heartbeat-style completion check
r = call("/v1/completions", {"model": M, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}); out("/v1/completions (heartbeat style)", bool(r["choices"][0]["text"]), repr(r["choices"][0]["text"]))
# 6 concurrent chats
def one(i):
    t0 = time.time(); r = chat({"max_tokens": 150, "temperature": 0, "messages": [{"role": "user", "content": "Write a short paragraph about topic number %d: the ocean." % i}]}); return r["usage"]["completion_tokens"], time.time() - t0
t0 = time.time()
with cf.ThreadPoolExecutor(6) as ex: res = list(ex.map(one, range(6)))
wall = time.time() - t0; toks = sum(x[0] for x in res)
out("6 concurrent requests", all(x[0] > 20 for x in res), "%d tokens in %.1fs = %.0f tok/s aggregate" % (toks, wall, toks / wall))
