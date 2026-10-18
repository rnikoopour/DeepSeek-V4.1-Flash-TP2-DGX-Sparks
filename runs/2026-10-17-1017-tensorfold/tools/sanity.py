#!/usr/bin/env python3
"""sanity.py BASE [MODEL]: chat, streaming chat, tool call (and tool-result round trip), thinking on, and a PNG image through /v1/chat/completions."""
import base64, json, struct, sys, time, urllib.request, zlib
base = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4.1-flash"
def post(body, timeout=300):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
def show(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name, extra, flush=True)
r = post({"model": model, "messages": [{"role": "user", "content": "What is 17 times 23? Answer with just the number."}], "max_tokens": 32, "temperature": 0})
c = r["choices"][0]["message"]["content"]; show("chat", "391" in c, repr(c))
r = post({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Say hello in French, one word."}], "max_tokens": 16, "temperature": 0})
show("alias glm-5.3-flash", bool(r["choices"][0]["message"]["content"]), repr(r["choices"][0]["message"]["content"]) + " model=" + r["model"])
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's the weather in Paris right now? Use celsius."}]
r = post({"model": model, "messages": msgs, "tools": tools, "max_tokens": 200, "temperature": 0})
m = r["choices"][0]["message"]; tc = m.get("tool_calls") or []
ok = len(tc) == 1 and tc[0]["function"]["name"] == "get_weather" and "paris" in tc[0]["function"]["arguments"].lower()
show("tool call", ok, json.dumps(tc)[:200] + " finish=" + r["choices"][0]["finish_reason"])
if tc:
    msgs2 = msgs + [m, {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temp_c": 14, "sky": "cloudy"})}]
    r = post({"model": model, "messages": msgs2, "tools": tools, "max_tokens": 120, "temperature": 0})
    c = r["choices"][0]["message"]["content"] or ""
    show("tool result round trip", "14" in c, repr(c[:120]))
r = post({"model": model, "messages": [{"role": "user", "content": "How many r letters are in strawberry?"}], "max_tokens": 600, "temperature": 0, "chat_template_kwargs": {"thinking": True, "enable_thinking": True}})
m = r["choices"][0]["message"]
show("thinking on", bool(m.get("reasoning_content") or m.get("reasoning")), "reasoning chars=%d content=%r" % (len(m.get("reasoning_content") or m.get("reasoning") or ""), (m.get("content") or "")[:60]))
# a 224x224 solid red PNG built by hand
def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
img = "data:image/png;base64," + base64.b64encode(png(448, 448, (220, 20, 20))).decode()
r = post({"model": model, "messages": [{"role": "user", "content": [{"type": "text", "text": "What color is this image? One word."}, {"type": "image_url", "image_url": {"url": img}}]}], "max_tokens": 16, "temperature": 0})
c = r["choices"][0]["message"]["content"]; show("image input", "red" in c.lower(), repr(c))
