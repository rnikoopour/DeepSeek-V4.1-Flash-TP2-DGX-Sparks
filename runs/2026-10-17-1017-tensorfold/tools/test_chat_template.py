#!/usr/bin/env python3
"""Check chat_template.jinja against DeepSeek's reference encoder (encoding/encoding.py of DeepSeek-V4.1).

    python3 test_chat_template.py --encoding /path/to/DeepSeek-V4.1-Flash-1017/encoding \
        [--template ../launch/chat_template.jinja] [--fuzz 3000] [--seed 1] [-v]

The template is rendered with Jinja2 set up the way Hugging Face's apply_chat_template sets it up
(ImmutableSandboxedEnvironment, loopcontrols, tojson = json.dumps with ensure_ascii=False, raise_exception),
with trim_blocks and lstrip_blocks on (HF's setting, TensorFold's lexer) and again with both off, and the text it
renders is compared byte for byte with encoding.encode_messages. The messages first pass through a model of
TensorFold's server-side normalization (server/messages.zig: leading system text merged, content parts flattened,
null content as "", tool-call arguments parsed into objects), because the template sees them that way.

Exit status 0 only if every case is byte-exact.

TensorFold's own Jinja engine (core/template, Zig) can be checked with the same cases:
    --zig-corpus corpus.json     writes a corpus for core/template/parity.zig (expected text = the reference's output)
    --zig-check corpus.json results.json   compares what that program rendered with the expected text
"""
import argparse
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path

import jinja2
import jinja2.ext
import jinja2.exceptions
from jinja2.sandbox import ImmutableSandboxedEnvironment

IMAGE = "<｜deepseek_image｜>"
HEADERS = ("<｜Assistant｜><think>", "<｜Assistant｜></think>")


# ---------------------------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------------------------

def make_env(blocks: bool) -> jinja2.Environment:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=blocks, lstrip_blocks=blocks, extensions=[jinja2.ext.loopcontrols]
    )

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def load_encoding(path: str):
    p = Path(path)
    if p.is_dir():
        p = p / "encoding.py"
    spec = importlib.util.spec_from_file_location("ds_encoding", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------------
# the server's view of the messages (server/messages.zig normalize + toolArguments)
# ---------------------------------------------------------------------------------------------------------------

def server_normalize(messages):
    """Leading system/developer text merged, later ones as system; content parts flattened; null content as "";
    tool-call arguments as objects (a string that is not a JSON object under _invalid_arguments)."""
    has_images = any(
        isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
        for m in messages
    )
    out, instructions = [], []
    for m in messages:
        m = copy.deepcopy(m)
        content = m.get("content")
        if isinstance(content, list):
            if has_images:
                parts = [IMAGE if p.get("type") == "image_url" else p.get("text") for p in content]
                m["content"] = "\n\n".join(parts)
            else:
                m["content"] = "".join(p.get("text") for p in content)
        elif content is None:
            m["content"] = ""
        if m.get("role") in ("system", "developer"):
            if not out:
                instructions.append(m)
                continue
            m["role"] = "system"
        out.append(m)
    if instructions:
        first = instructions[0]
        first["role"] = "system"
        first["content"] = "\n\n".join(x["content"] for x in instructions)
        out.insert(0, first)
    for m in out:
        calls = m.get("tool_calls")
        if not (isinstance(calls, list) and calls):
            continue
        for call in calls:
            fn = call.get("function")
            if not isinstance(fn, dict) or "arguments" not in fn:
                continue
            args = fn["arguments"]
            if isinstance(args, dict):
                continue
            parsed = None
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except ValueError:
                    parsed = None
            if not isinstance(parsed, dict):
                parsed = {"_invalid_arguments": args}
            fn["arguments"] = parsed
    return out


# ---------------------------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------------------------

class Runner:
    def __init__(self, enc, template_path: Path):
        self.enc = enc
        self.source = template_path.read_text(encoding="utf-8")
        self.compiled = {}
        for blocks in (True, False):
            self.compiled[blocks] = make_env(blocks).from_string(self.source)

    def render(self, messages, tools, mode, effort, gen=True, blocks=True, normalize=True):
        msgs = server_normalize(messages) if normalize else copy.deepcopy(messages)
        ctx = dict(
            messages=msgs,
            tools=copy.deepcopy(tools) if tools else None,
            add_generation_prompt=gen,
            enable_thinking=(mode == "thinking"),
            thinking_mode=mode,
        )
        if effort is not None:
            ctx["reasoning_effort"] = effort
        return self.compiled[blocks].render(**ctx)

    def reference(self, messages, tools, mode, effort):
        msgs = copy.deepcopy(messages)
        if tools:
            # the template puts variable-given tools on the first system message and adds an empty one if absent
            if not msgs or msgs[0].get("role") != "system":
                msgs.insert(0, {"role": "system", "content": "", "tools": copy.deepcopy(tools)})
            else:
                msgs[0]["tools"] = copy.deepcopy(tools)
        return self.enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)


def first_diff(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    lo = max(0, i - 40)
    return f"first difference at char {i}: expected ...{a[lo:i + 60]!r}  got ...{b[lo:i + 60]!r}"


# ---------------------------------------------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------------------------------------------

def call(name, args, cid=None, **extra):
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid is not None:
        c["id"] = cid
    c.update(extra)
    return c


def tool_def(name, desc="d", props=None, **extra):
    fn = {"name": name, "description": desc,
          "parameters": {"type": "object", "properties": props or {"q": {"type": "string"}}, "required": list((props or {"q": 0}))}}
    t = {"type": "function", "function": fn}
    t.update(extra)
    return t


W = tool_def("get_weather", "Get the weather for a city", {"location": {"type": "string", "description": "The city"},
                                                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}})
S = tool_def("search", "Search the web: 中文 ✓ \"quoted\" and 'single'", {"query": {"type": "string"}, "n": {"type": "integer"},
                                                                       "tags": {"type": "array", "items": {"type": "string"}}})
SYS = {"role": "system", "content": "You are a helpful assistant."}
U1 = {"role": "user", "content": "Hello"}


def tool_turns():
    """A tool conversation, several variants of how its results arrive."""
    calls3 = [call("get_weather", {"location": "Beijing"}, "c1"), call("search", {"query": "rain", "n": 3}, "c2"),
              call("get_weather", '{"location": "Paris", "unit": "celsius"}', "c3")]
    base = [SYS, {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking", "reasoning_content": "need tools", "tool_calls": calls3}]
    t = lambda i, c: {"role": "tool", "tool_call_id": i, "content": c}
    return {
        "in_order": base + [t("c1", "r1"), t("c2", "r2"), t("c3", "r3")],
        "shuffled": base + [t("c3", "r3"), t("c1", "r1"), t("c2", "r2")],
        "unknown_ids": base + [t("zz", "rz"), t("c2", "r2"), t("c1", "r1"), t("", "re")],
        "dup_ids": base + [t("c2", "a"), t("c1", "b"), t("c2", "c")],
        "no_ids": base + [{"role": "tool", "content": "x"}, {"role": "tool", "content": "y"}],
    }


def cases():
    out = []

    def add(name, messages, mode="chat", tools=None, effort=None, **kw):
        out.append(dict(name=name, messages=messages, mode=mode, tools=tools, effort=effort, **kw))

    for mode in ("chat", "thinking"):
        add(f"user only [{mode}]", [U1], mode)
        add(f"system+user [{mode}]", [SYS, U1], mode)
        add(f"system only [{mode}]", [SYS], mode)
        add(f"no messages [{mode}]", [], mode)
        add(f"multi-turn, ends with user [{mode}]", [SYS, U1, {"role": "assistant", "content": "Hi!", "reasoning_content": "greet"},
                                                    {"role": "user", "content": "Capital of France?"}], mode)
        add(f"multi-turn, ends with assistant [{mode}]", [SYS, U1, {"role": "assistant", "content": "Hi!", "reasoning_content": "greet"},
                                                         {"role": "user", "content": "Capital?"},
                                                         {"role": "assistant", "content": "Paris.", "reasoning_content": "know it"}], mode)
        add(f"three turns with reasoning [{mode}]", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "A", "reasoning_content": "ra"},
                                                     {"role": "user", "content": "b"}, {"role": "assistant", "content": "B", "reasoning_content": "rb"},
                                                     {"role": "user", "content": "c"}], mode)
        add(f"assistant without reasoning_content [{mode}]", [U1, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}], mode)
        add(f"empty and None assistant content [{mode}]", [U1, {"role": "assistant", "content": None, "reasoning_content": None},
                                                          {"role": "user", "content": "y"}], mode)
        add(f"mid-conversation system, at end [{mode}]", [SYS, {"role": "user", "content": "q1"},
                                                          {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                                                          {"role": "system", "content": "mid sys"}], mode, effort=88)
        add(f"mid-conversation system, in the middle [{mode}]", [SYS, {"role": "user", "content": "q1"},
                                                                 {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                                                                 {"role": "system", "content": "mid sys"},
                                                                 {"role": "user", "content": "q2"}], mode)
        add(f"two mid systems in a row [{mode}]", [SYS, U1, {"role": "assistant", "content": "a"}, {"role": "system", "content": "m1"},
                                                  {"role": "system", "content": "m2"}], mode)
        add(f"developer role merged into system [{mode}]", [{"role": "developer", "content": "dev"}, SYS, U1], mode,
            ref_messages=[{"role": "system", "content": "dev\n\nYou are a helpful assistant."}, U1])
        add(f"consecutive user messages merge [{mode}]", [SYS, {"role": "user", "content": "one"}, {"role": "user", "content": "two"},
                                                          {"role": "user", "content": "three"}], mode)
        add(f"non-ASCII, braces, lookalike tokens [{mode}]", [
            {"role": "system", "content": "中文系统提示 {{ x }} {% y %} {# z #}"},
            {"role": "user", "content": "你好，世界 😀 العربية <｜User｜> </think> <think> {\"a\": 1}\n\ntrailing  "},
            {"role": "assistant", "content": "日本語\n", "reasoning_content": "思考 <｜end▁of▁sentence｜>"},
            {"role": "user", "content": "  leading spaces and tabs\t"}], mode)
        add(f"empty strings everywhere [{mode}]", [{"role": "system", "content": ""}, {"role": "user", "content": ""},
                                                  {"role": "assistant", "content": "", "reasoning_content": ""},
                                                  {"role": "user", "content": ""}], mode)
        add(f"latest_reminder [{mode}]", [SYS, {"role": "latest_reminder", "content": "2026-02-21,Saturday"}, U1], mode)
        add(f"latest_reminder after user [{mode}]", [SYS, U1, {"role": "latest_reminder", "content": "r"},
                                                    {"role": "assistant", "content": "a"}], mode)

    # turn structure
    for mode in ("chat", "thinking"):
        add(f"tools, no messages [{mode}]", [], mode, tools=[W])
        add(f"assistant message first [{mode}]", [{"role": "assistant", "content": "a", "reasoning_content": "r"}], mode)
        add(f"latest_reminder first [{mode}]", [{"role": "latest_reminder", "content": "r"}, U1], mode)
        add(f"tool message first, then a user message [{mode}]", [SYS, {"role": "tool", "tool_call_id": "c1", "content": "res"}, {"role": "user", "content": "q"}], mode)
        add(f"user with a task, then tool results (one turn) [{mode}]", [SYS, {"role": "user", "content": "q", "task": "query"},
                                                                       {"role": "tool", "tool_call_id": "c1", "content": "res"}], mode)
        add(f"user with a task, tool result, user (two turns) [{mode}]", [SYS, {"role": "user", "content": "q", "task": "query"},
                                                                         {"role": "tool", "tool_call_id": "c1", "content": "res"}, {"role": "user", "content": "b"}], mode)
        add(f"task user, user, task user [{mode}]", [SYS, {"role": "user", "content": "a", "task": "query"}, {"role": "user", "content": "b"},
                                                    {"role": "user", "content": "c", "task": "domain"}, {"role": "assistant", "content": "x", "reasoning_content": "y"}], mode)
        add(f"mid-conversation system after tool results [{mode}]", [SYS, {"role": "user", "content": "q"},
                                                                    {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [call("a", {}, "c1")]},
                                                                    {"role": "tool", "tool_call_id": "c1", "content": "res"}, {"role": "system", "content": "note"}], mode, tools=[W])
        add(f"tool results only after the last user turn [{mode}]", [SYS, {"role": "user", "content": "q"}, {"role": "assistant", "content": "x", "reasoning_content": "r"},
                                                                    {"role": "user", "content": "q2"},
                                                                    {"role": "assistant", "content": "", "reasoning_content": "r2", "tool_calls": [call("a", {}, "c1")]},
                                                                    {"role": "tool", "tool_call_id": "c1", "content": "res"},
                                                                    {"role": "assistant", "content": "ok", "reasoning_content": "r3"}], mode)
        add(f"long conversation [{mode}]", [SYS] + [m for k in range(60) for m in (
            {"role": "user", "content": f"question {k}"},
            {"role": "assistant", "content": f"answer {k}", "reasoning_content": f"thought {k}", "tool_calls": [call("a", {"k": k}, f"id{k}")]},
            {"role": "tool", "tool_call_id": f"id{k}", "content": f"result {k}"},
            {"role": "assistant", "content": f"done {k}", "reasoning_content": f"more {k}"})], mode)

    # reasoning effort
    for effort in (None, "low", "high", "max", 1, 30, 42, 75, 100):
        add(f"effort {effort!r}, system+user [thinking]", [SYS, U1], "thinking", effort=effort)
        add(f"effort {effort!r}, user only [thinking]", [U1], "thinking", effort=effort)
        add(f"effort {effort!r} is ignored [chat]", [SYS, U1], "chat", effort=effort)
    for effort in ("medium", 0, 101, -1, True, 1.5, "LOW"):
        add(f"invalid effort {effort!r} [thinking]", [SYS, U1], "thinking", effort=effort, expect_error=True)
    add("invalid thinking_mode", [U1], "bogus", expect_error=True)

    # tools in the system prompt
    for mode in ("chat", "thinking"):
        add(f"tools, system+user [{mode}]", [SYS, U1], mode, tools=[W, S])
        add(f"tools, one tool [{mode}]", [SYS, U1], mode, tools=[W])
        add(f"tools without a system message [{mode}]", [U1], mode, tools=[W, S])
        add(f"tools, multi-turn keeps all reasoning [{mode}]", [SYS, U1, {"role": "assistant", "content": "a", "reasoning_content": "r1"},
                                                              {"role": "user", "content": "b"}, {"role": "assistant", "content": "c", "reasoning_content": "r2"},
                                                              {"role": "user", "content": "d"}], mode, tools=[W])
        add(f"tools, mid-conversation system [{mode}]", [SYS, U1, {"role": "assistant", "content": "a", "reasoning_content": "r"},
                                                       {"role": "system", "content": "mid"}], mode, tools=[W])
        add(f"tools declared on a mid-conversation system message [{mode}]", [SYS, U1, {"role": "assistant", "content": "a", "reasoning_content": "r"},
                                                                              {"role": "system", "content": "mid", "tools": [S]}, {"role": "user", "content": "q"}], mode,
            ref_messages=None)
        add(f"tools with schema keys named items/keys/get [{mode}]", [SYS, U1], mode, tools=[tool_def(
            "odd", "d", {"items": {"type": "array", "items": {"type": "string"}}, "keys": {"type": "string"}, "get": {"type": "string"}})])
        add(f"tool with unicode and escapes [{mode}]", [SYS, U1], mode, tools=[tool_def("t", "描述 \"q\" \\ \n tab\t é", {"é": {"type": "string", "description": "😀"}})])
        add(f"tool function without description [{mode}]", [SYS, U1], mode, tools=[{"type": "function", "function": {"name": "bare"}}])
        add(f"tool function key order is kept [{mode}]", [SYS, U1], mode, tools=[{"type": "function", "function": {
            "parameters": {"type": "object", "properties": {}}, "description": "late description", "name": "reordered"}}])

    # namespaces
    ns_str = tool_def("lookup", "Look up a value", None, namespace="search")
    ns_obj = tool_def("lookup", "Look up a value", None, namespace={"name": "search", "description": "Search tools."})
    ns_fn = tool_def("lookup", "Look up a value")
    ns_fn["function"]["namespace"] = "search"
    ns_fn_obj = tool_def("lookup", "Look up a value")
    ns_fn_obj["function"]["namespace"] = {"name": "search", "description": "Search tools."}
    ns_nodesc = {"type": "function", "namespace": {"name": "files", "description": "File tools."}, "function": {"name": "read"}}
    qualified = tool_def("search::lookup", "Look up a value", None, namespace={"name": "search", "description": "Search tools."})
    plain = tool_def("lookup", "Look up a value")
    for nm, tl in (("namespace string on tool", ns_str), ("namespace object on tool", ns_obj), ("namespace string on function", ns_fn),
                   ("namespace object on function", ns_fn_obj), ("namespace object, function without description", ns_nodesc),
                   ("qualified name plus namespace", qualified)):
        add(f"tools: {nm}", [SYS, U1], "chat", tools=[tl])
    add("tools: same name in separate namespaces", [SYS, U1], "chat", tools=[plain, ns_str, tool_def("lookup", "x", None, namespace={"name": "files"})])
    add("tools: conflicting namespaces", [SYS, U1], "chat", tools=[tool_def("a::b", "d", None, namespace="c")], expect_error=True)
    add("tools: nested :: in name", [SYS, U1], "chat", tools=[tool_def("a::b::c", "d")], expect_error=True)
    add("tools: :: in namespace", [SYS, U1], "chat", tools=[tool_def("b", "d", None, namespace="a::x")], expect_error=True)

    # assistant tool calls
    for mode in ("chat", "thinking"):
        msgs = lambda calls, content="": [SYS, {"role": "user", "content": "q"}, {"role": "assistant", "content": content,
                                                                                  "reasoning_content": "because", "tool_calls": calls}]
        add(f"tool call, JSON string arguments [{mode}]", msgs([call("get_weather", '{"location": "Beijing", "unit": "celsius"}', "c1")]), mode, tools=[W])
        add(f"tool call, dict arguments [{mode}]", msgs([call("get_weather", {"location": "Beijing", "unit": "celsius"}, "c1")]), mode, tools=[W])
        add(f"tool call, mixed value types [{mode}]", msgs([call("t", {
            "s": "text\nwith \"quotes\" 中文", "i": 3, "f": 1.5, "f2": 100.0, "e": 1e-7, "big": 12345678901234567890, "neg": -4, "b": True,
            "b2": False, "n": None, "l": [1, "a", None, {"k": [2]}], "d": {"é": "ü", "x": {"y": []}}, "empty_s": "", "empty_l": [], "empty_d": {}}, "c1")]), mode)
        add(f"tool call, several calls [{mode}]", msgs([call("a", {"x": 1}, "c1"), call("b", '{"y": "two"}', "c2"), call("c", {}, "c3")]), mode)
        add(f"tool call, no arguments [{mode}]", msgs([call("a", {}, "c1")]), mode)
        add(f"tool call, empty-string arguments [{mode}]", msgs([call("a", "", "c1")]), mode)
        add(f"tool call only, empty content [{mode}]", msgs([call("a", {"x": 1}, "c1")], ""), mode)
        add(f"tool call only, None content [{mode}]", msgs([call("a", {"x": 1}, "c1")], None), mode)
        add(f"tool call with text content [{mode}]", msgs([call("a", {"x": 1}, "c1")], "I will look it up.\n"), mode)
        add(f"tool calls: [] [{mode}]", msgs([], "just text"), mode)
        add(f"tool call arguments not JSON [{mode}]", msgs([call("a", "not json", "c1")]), mode)
        add(f"tool call arguments a JSON array [{mode}]", msgs([call("a", "[1, 2]", "c1")]), mode)
        add(f"tool call arguments a JSON scalar [{mode}]", msgs([call("a", "42", "c1")]), mode)
        add(f"tool call, namespace on the call [{mode}]", msgs([call("lookup", {"q": "v"}, "c1", namespace="search")]), mode)
        add(f"tool call, namespace on the function [{mode}]", msgs([{"type": "function", "id": "c1", "function": {"name": "lookup", "namespace": "search", "arguments": {"q": 1}}}]), mode)
        add(f"tool call, qualified name [{mode}]", msgs([call("search::lookup", {"q": "v"}, "c1")]), mode)
        add(f"tool call, qualified name and namespace agree [{mode}]", msgs([call("search::lookup", {"q": "v"}, "c1", namespace="search")]), mode)
        add(f"tool call, qualified name and namespace clash [{mode}]", msgs([call("search::lookup", {"q": "v"}, "c1", namespace="files")]), mode, expect_error=True)
        add(f"tool call without an id [{mode}]", msgs([call("a", {"x": 1})]), mode)
    add("tool call arguments double-encoded JSON (server wraps them)", [SYS, U1, {"role": "assistant", "content": "", "tool_calls": [
        call("a", json.dumps(json.dumps({"x": 1})), "c1")]}], "chat", known_gap=True)

    # tool results
    for mode in ("chat", "thinking"):
        for label, conv in tool_turns().items():
            add(f"tool results {label}, ends with results [{mode}]", conv, mode, tools=[W, S])
            add(f"tool results {label}, then assistant [{mode}]", conv + [{"role": "assistant", "content": "done", "reasoning_content": "ok"}], mode, tools=[W, S])
            add(f"tool results {label}, then user [{mode}]", conv + [{"role": "user", "content": "thanks, and Rome?"}], mode, tools=[W, S])
            add(f"tool results {label}, no tools declared [{mode}]", conv, mode)
            add(f"tool results {label}, user text first [{mode}]", conv[:3] + [{"role": "user", "content": "extra context"}] + conv[3:], mode, tools=[W, S])
        c1 = [call("get_weather", {"location": "A"}, "c1")]
        base = [SYS, {"role": "user", "content": "q"}, {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": c1}]
        add(f"one tool result [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": "sunny"}], mode, tools=[W])
        add(f"one tool result, followed by a user message [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                                                                          {"role": "user", "content": "and tomorrow?"}], mode, tools=[W])
        add(f"tool result, a second tool round [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                                                                {"role": "assistant", "content": "", "reasoning_content": "more", "tool_calls": [call("search", {"query": "x"}, "c9")]},
                                                                {"role": "tool", "tool_call_id": "c9", "content": "found"}], mode, tools=[W, S])
        add(f"tool result JSON content [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": "{\"t\": 22, \"u\": \"°C\"}"}], mode, tools=[W])
        add(f"tool result empty content [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": ""}], mode, tools=[W])
        add(f"tool result None content [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": None}], mode, tools=[W], ref_messages=base + [{"role": "tool", "tool_call_id": "c1", "content": ""}])
        add(f"tool result multi-line, unicode [{mode}]", base + [{"role": "tool", "tool_call_id": "c1", "content": "第一行\n第二行\n\n  indented {x}"}], mode, tools=[W])
        add(f"tool message before any call [{mode}]", [SYS, {"role": "tool", "tool_call_id": "c1", "content": "orphan"}, {"role": "user", "content": "q"}], mode)
        add(f"tool results from the earlier round keep sorting by the latest calls [{mode}]", base + [
            {"role": "tool", "tool_call_id": "c1", "content": "r"}, {"role": "assistant", "content": "x", "reasoning_content": "y"},
            {"role": "user", "content": "again"}, {"role": "assistant", "content": "", "reasoning_content": "z", "tool_calls": [call("a", {}, "d1"), call("b", {}, "d2")]},
            {"role": "tool", "tool_call_id": "d2", "content": "second"}, {"role": "tool", "tool_call_id": "d1", "content": "first"}], mode, tools=[W])
        add(f"order stays from the last assistant that had calls [{mode}]", [SYS, {"role": "user", "content": "q"},
                                                                         {"role": "assistant", "content": "", "tool_calls": [call("a", {}, "e1"), call("b", {}, "e2")]},
                                                                         {"role": "assistant", "content": "no calls here"},
                                                                         {"role": "tool", "tool_call_id": "e2", "content": "2"}, {"role": "tool", "tool_call_id": "e1", "content": "1"}], mode)

    # tasks (quick instructions)
    for task in ("action", "query", "authority", "domain", "title", "read_url"):
        for mode in ("chat", "thinking"):
            add(f"task {task} on the last user message [{mode}]", [SYS, {"role": "user", "content": "q", "task": task}], mode)
            add(f"task {task} on a user message before an assistant reply [{mode}]", [SYS, {"role": "user", "content": "q", "task": task},
                                                                                    {"role": "assistant", "content": "answer"}], mode)
    add("task title on an assistant message, with title", [SYS, U1, {"role": "assistant", "content": "Hi", "reasoning_content": "r", "task": "title"},
                                                           {"role": "assistant", "content": "A title", "reasoning_content": "t"}], "thinking")
    add("task on a user message stops the merge", [SYS, {"role": "user", "content": "a", "task": "query"}, {"role": "user", "content": "b"}], "chat")
    add("invalid task", [SYS, {"role": "user", "content": "q", "task": "nope"}], "chat", expect_error=True)
    add("assistant without eos", [SYS, U1, {"role": "assistant", "content": "partial", "wo_eos": True}], "chat")

    # content as a list of parts (the template flattens them itself when the server did not)
    add("user content parts (template joins with a blank line)", [SYS, {"role": "user", "content": [
        {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}], "chat", raw=True)
    add("user content parts with images", [SYS, {"role": "user", "content": [
        {"type": "text", "text": "第一张图"}, {"type": "image_url", "image_url": {"url": "x.png"}}, {"type": "text", "text": "和第二张图"}]}], "thinking", raw=True,
        ref_messages=None)
    add("system and assistant content parts", [{"role": "system", "content": [{"type": "text", "text": "s1"}, {"type": "text", "text": "s2"}]}, U1,
                                               {"role": "assistant", "content": [{"type": "text", "text": "a1"}, {"type": "text", "text": "a2"}]}, {"role": "user", "content": "q"}],
        "chat", raw=True)
    add("tool result content parts", [SYS, {"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [call("a", {}, "c1")]},
                                      {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "t1"}, {"type": "text", "text": "t2"}]}], "chat", raw=True)
    tr = lambda i, t="res": {"type": "tool_result", "tool_use_id": i, "content": t}
    asst = {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [call("a", {}, "c1"), call("b", {}, "c2")]}
    for mode in ("chat", "thinking"):
        add(f"user content of only tool_result parts is not a user turn [{mode}]", [SYS, {"role": "user", "content": "q"}, asst,
                                                                                 {"role": "user", "content": [tr("c2", "two"), tr("c1", "one")]}], mode, raw=True)
        add(f"user content of tool_result and text parts [{mode}]", [SYS, {"role": "user", "content": "q"}, asst,
                                                                    {"role": "user", "content": [tr("c2", "two"), {"type": "text", "text": "and"}, tr("c1", "one")]}], mode, raw=True)
        add(f"tool message with a task field is still a tool result [{mode}]", [SYS, U1, asst,
                                                                              {"role": "tool", "tool_call_id": "c1", "content": "x", "task": "query"}], mode)
    add("text-only content parts through the server (concatenated, a known server/reference difference)", [SYS, {"role": "user", "content": [
        {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}], "chat", known_gap=True)
    return out


# ---------------------------------------------------------------------------------------------------------------
# official cases
# ---------------------------------------------------------------------------------------------------------------

def official(enc, runner: Runner, enc_dir: Path):
    results = []
    tests = enc_dir / "tests"
    for inp in sorted(tests.glob("test_input_*.json")):
        n = inp.stem.split("_")[-1]
        want = (tests / f"test_output_{n}.txt").read_text()
        data = json.loads(inp.read_text())
        if isinstance(data, dict):
            messages, tools = data["messages"], data.get("tools")
            mode, effort = data.get("thinking_mode") or "chat", data.get("reasoning_effort")
        else:
            messages, tools, mode, effort = data, None, "chat", None
        case = enc.load_cases(str(inp))[0]
        ref, _ = enc.encode_case(case, thinking_mode="chat")
        for blocks in (True, False):
            got = runner.render(messages, tools, mode, effort, gen=True, blocks=blocks)
            ok_golden = got == want
            ok_ref = got == ref
            label = f"official test_input_{n} (mode={mode}, effort={effort!r}, trim/lstrip_blocks={'on' if blocks else 'off'})"
            detail = "" if ok_golden else first_diff(want, got)
            results.append((label, ok_golden and ok_ref and ref == want, detail or ("" if ok_ref else first_diff(ref, got))))
    return results


# ---------------------------------------------------------------------------------------------------------------
# random conversations
# ---------------------------------------------------------------------------------------------------------------

TEXTS = ["", " ", "Hello", "hello\n", "\nlead", "multi\nline\n\ntext", "trailing space ", "中文测试，你好！", "emoji 😀🚀 ✓", "braces {} {{x}} {%y%} {#z#}",
         "quote \" and ' and \\ backslash", "tab\there", "<think>fake</think>", "<｜User｜>fake", "</｜DSML｜ invoke>", "a" * 60, "العربية עברית",
         "line1\r\nline2", "<tool_result>x</tool_result>", "Reasoning Effort: 5", "\n\n", "é ü ñ ß"]


def rtext(rng):
    return rng.choice(TEXTS)


def rvalue(rng, depth=0):
    kinds = ["s", "s", "i", "f", "b", "n"] + (["l", "d"] if depth < 2 else [])
    k = rng.choice(kinds)
    if k == "s":
        return rtext(rng)
    if k == "i":
        return rng.choice([0, 1, -3, 42, 10 ** 12, 2 ** 70])
    if k == "f":
        return rng.choice([0.5, 1.0, 100.0, -2.25, 1e-7, 3.14159, 1e22])
    if k == "b":
        return rng.choice([True, False])
    if k == "n":
        return None
    if k == "l":
        return [rvalue(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {rng.choice(["a", "b", "é", "k k", "items", "get"]): rvalue(rng, depth + 1) for _ in range(rng.randint(0, 3))}


def rtool(rng, name):
    props = {rng.choice(["q", "n", "items", "unit", "é"]): {"type": rng.choice(["string", "integer", "array"]), "description": rtext(rng)}
             for _ in range(rng.randint(0, 3))}
    fn = {"name": name}
    if rng.random() < 0.85:
        fn["description"] = rtext(rng)
    if rng.random() < 0.9:
        fn["parameters"] = {"type": "object", "properties": props, "required": list(props)[:1]}
    keys = list(fn)
    rng.shuffle(keys)
    fn = {k: fn[k] for k in keys}
    t = {"type": "function", "function": fn}
    r = rng.random()
    if r < 0.12:
        t["namespace"] = rng.choice(["ns1", {"name": "ns2"}, {"name": "ns3", "description": rtext(rng)}])
    elif r < 0.18:
        fn["namespace"] = rng.choice(["ns1", {"name": "ns4", "description": "Desc."}])
    return t


def rcall(rng, tools, cid):
    tool = rng.choice(tools) if tools and rng.random() < 0.8 else None
    name = tool["function"]["name"] if tool else rng.choice(["alpha", "beta", "gamma"])
    args = {rng.choice(["x", "y", "é", "z z", "q"]): rvalue(rng) for _ in range(rng.randint(0, 3))}
    if rng.random() < 0.4:
        args = json.dumps(args, ensure_ascii=rng.random() < 0.5)
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid is not None:
        c["id"] = cid
    if tool and rng.random() < 0.6:
        ns = tool.get("namespace") or tool["function"].get("namespace")
        if ns is not None and rng.random() < 0.7:
            c["namespace"] = ns["name"] if isinstance(ns, dict) else ns
    return c


def rcase(rng):
    mode = rng.choice(["chat", "thinking"])
    effort = rng.choice([None, "low", "high", "max", 1, 30, 75, 100, rng.randint(1, 100)])
    tools = None
    if rng.random() < 0.4:
        names = rng.sample(["get_weather", "search", "lookup", "calc", "open"], rng.randint(1, 3))
        tools = [rtool(rng, n) for n in names]
    msgs = []
    if rng.random() < 0.8:
        msgs.append({"role": "system", "content": rtext(rng)})
    if rng.random() < 0.08:
        msgs.append({"role": "latest_reminder", "content": rtext(rng)})
    counter = [0]

    def new_id():
        counter[0] += 1
        return f"call_{counter[0]}"

    def user():
        m = {"role": "user", "content": rtext(rng)}
        if rng.random() < 0.04:
            m["task"] = rng.choice(["action", "query", "authority", "domain", "read_url"])
        return m

    def assistant(allow_calls=True):
        m = {"role": "assistant", "content": rng.choice([None, rtext(rng), rtext(rng), ""])}
        if rng.random() < 0.7:
            m["reasoning_content"] = rng.choice([rtext(rng), None, ""])
        calls = []
        if allow_calls and rng.random() < 0.45:
            for _ in range(rng.randint(1, 3)):
                calls.append(rcall(rng, tools, new_id() if rng.random() < 0.93 else rng.choice([None, ""])))
            m["tool_calls"] = calls
        elif rng.random() < 0.05:
            m["tool_calls"] = []
        if rng.random() < 0.03:
            m["task"] = "title"
        if rng.random() < 0.02:
            m["wo_eos"] = True
        return m, calls

    def tool_results(calls):
        ids = [c.get("id") for c in calls]
        if rng.random() < 0.3:
            rng.shuffle(ids)
        res = []
        for i in ids:
            if rng.random() < 0.9:
                res.append({"role": "tool", "tool_call_id": i if i is not None else "", "content": rtext(rng)})
        if rng.random() < 0.1:
            res.append({"role": "tool", "tool_call_id": rng.choice(["zzz", "", ids[0] if ids else "q"]), "content": rtext(rng)})
        if res and rng.random() < 0.15:
            res.insert(rng.randint(0, len(res)), user())
        return res

    for _ in range(rng.randint(0, 4)):
        if rng.random() < 0.1 and any(m["role"] != "system" for m in msgs):  # a leading one would be merged by the server
            msgs.append({"role": "system", "content": rtext(rng)})
        msgs.append(user())
        while rng.random() < 0.15:
            msgs.append(user())
        rounds = 0
        while rng.random() < 0.9 and rounds < 3:
            a, calls = assistant()
            msgs.append(a)
            rounds += 1
            if calls and rng.random() < 0.9:
                msgs.extend(tool_results(calls))
                if rng.random() < 0.6:
                    continue
            break
    if msgs and rng.random() < 0.5:
        msgs = msgs[: rng.randint(1, len(msgs))]
    return dict(messages=msgs, tools=tools, mode=mode, effort=effort)


# ---------------------------------------------------------------------------------------------------------------
# corpus for TensorFold's own template engine (core/template/parity.zig)
# ---------------------------------------------------------------------------------------------------------------

def write_zig_corpus(runner: Runner, enc, enc_dir: Path, path: str, fuzz: int, seed: int):
    """Every comparable case as a parity.zig corpus: messages as the template sees them, expected text from the reference."""
    conversations, tool_sets, contexts, out = [], [], [], []

    def add(name, messages, tools, mode, effort, gen, text):
        ctx = {"thinking_mode": mode, "enable_thinking": mode == "thinking"}
        if effort is not None:
            ctx["reasoning_effort"] = effort
        conversations.append(messages)
        if tools:
            tool_sets.append(tools)
        contexts.append(ctx)
        out.append({"name": name, "t": "ds", "m": len(conversations) - 1, "tl": (len(tool_sets) - 1) if tools else None,
                    "c": len(contexts) - 1, "g": gen, "ok": text is not None, "text": text or "", "raised": False, "error": ""})

    def from_case(c):
        if c.get("known_gap"):
            return
        normalize = not c.get("raw", False)
        seen = server_normalize(c["messages"]) if normalize else copy.deepcopy(c["messages"])
        ref_messages = c.get("ref_messages") or c["messages"]
        try:
            want = runner.reference(ref_messages, c["tools"], c["mode"], c["effort"])
        except Exception:  # noqa: BLE001
            want = None
        if c.get("expect_error") or want is None:
            if c.get("expect_error"):
                add(c["name"], seen, c["tools"], c["mode"], c["effort"], True, None)
            return
        add(c["name"], seen, c["tools"], c["mode"], c["effort"], True, want)
        if not any(m.get("task") is not None for m in ref_messages):
            last = ref_messages[-1] if ref_messages else {}
            if last.get("role") != "assistant":
                stripped = strip_header(want)
                add(c["name"] + " (no generation prompt)", seen, c["tools"], c["mode"], c["effort"], False, stripped if stripped is not None else want)

    for inp in sorted((enc_dir / "tests").glob("test_input_*.json")):
        n = inp.stem.split("_")[-1]
        data = json.loads(inp.read_text())
        if isinstance(data, dict):
            messages, tools = data["messages"], data.get("tools")
            mode, effort = data.get("thinking_mode") or "chat", data.get("reasoning_effort")
        else:
            messages, tools, mode, effort = data, None, "chat", None
        add(f"official test_input_{n}", server_normalize(messages), tools, mode, effort, True, (enc_dir / "tests" / f"test_output_{n}.txt").read_text())
    for c in cases():
        from_case(c)
    rng = random.Random(seed)
    for k in range(fuzz):
        c = rcase(rng)
        c["name"] = f"fuzz #{k}"
        from_case(c)
    Path(path).write_text(json.dumps({"templates": {"ds": runner.source}, "conversations": conversations, "tools": tool_sets,
                                      "contexts": contexts, "cases": out}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} cases to {path}")


def check_zig_results(corpus_path: str, results_path: str) -> int:
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    bad = 0
    for case, r in zip(corpus["cases"], results):
        if case["ok"]:
            good = r["ok"] and r["text"] == case["text"]
        else:
            good = not r["ok"]
        if not good:
            bad += 1
            if bad <= 10:
                detail = first_diff(case["text"], r["text"]) if r["ok"] else f"engine error: {r['error']}"
                print(f"ENGINE FAIL  {case['name']}: {detail}")
    print(f"engine: {len(results) - bad} of {len(results)} cases rendered byte-exact")
    return 1 if bad or len(results) != len(corpus["cases"]) else 0


# ---------------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------------

def strip_header(text: str):
    for h in HEADERS:
        if text.endswith(h):
            return text[: -len(h)]
    return None


def run_case(runner: Runner, c: dict):
    """Returns (status, detail): PASS, FAIL or GAP (a documented difference that really differs)."""
    expect_error = c.get("expect_error", False)
    normalize = not c.get("raw", False)
    ref_messages = c.get("ref_messages") or c["messages"]
    try:
        want = runner.reference(ref_messages, c["tools"], c["mode"], c["effort"])
        ref_err = None
    except Exception as e:  # noqa: BLE001
        want, ref_err = None, f"{type(e).__name__}: {e}"
    results = {}
    for blocks in (True, False):
        try:
            results[blocks] = (runner.render(c["messages"], c["tools"], c["mode"], c["effort"], gen=True, blocks=blocks, normalize=normalize), None)
        except Exception as e:  # noqa: BLE001
            results[blocks] = (None, f"{type(e).__name__}: {e}")
    if expect_error:
        ok = ref_err is not None and results[True][1] is not None and results[False][1] is not None
        return ("PASS" if ok else "FAIL", f"ref_error={ref_err!r} template_error={results[True][1]!r}")
    if ref_err is not None:
        return ("FAIL", f"reference raised {ref_err}")
    for blocks in (True, False):
        got, err = results[blocks]
        if err is not None:
            return ("FAIL", f"template raised {err}")
        if got != want:
            status = "GAP" if c.get("known_gap") else "FAIL"
            return (status, f"blocks={'on' if blocks else 'off'}: {first_diff(want, got)}")
    if c.get("known_gap"):
        return ("PASS", "matches (the documented difference does not show here)")
    # without the generation prompt the text is the same minus the trailing assistant header
    try:
        nogen = runner.render(c["messages"], c["tools"], c["mode"], c["effort"], gen=False, normalize=normalize)
    except Exception as e:  # noqa: BLE001
        return ("FAIL", f"template raised {e} with add_generation_prompt=false")
    last = (ref_messages[-1] if ref_messages else {})
    if any(m.get("task") is not None for m in ref_messages):
        # a task token ends the prompt only with add_generation_prompt; here just require the prefix relation
        if not want.startswith(nogen):
            return ("FAIL", "add_generation_prompt=false is not a prefix: " + first_diff(want[: len(nogen)], nogen))
    elif last.get("task") is None and last.get("role") != "assistant":
        stripped = strip_header(want)
        expected_nogen = stripped if stripped is not None else want
        if nogen != expected_nogen:
            return ("FAIL", "add_generation_prompt=false: " + first_diff(expected_nogen, nogen))
    elif last.get("role") == "assistant" and last.get("task") is None:
        if nogen != want:
            return ("FAIL", "add_generation_prompt=false: " + first_diff(want, nogen))
    return ("PASS", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoding", required=True, help="DeepSeek's encoding directory (or encoding.py)")
    ap.add_argument("--template", default=str(Path(__file__).resolve().parent.parent / "launch" / "chat_template.jinja"))
    ap.add_argument("--fuzz", type=int, default=0, help="also compare this many random conversations")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true", help="show details of passing cases too")
    ap.add_argument("--zig-corpus", metavar="OUT.json", help="write a corpus for TensorFold's template engine (parity.zig) and exit")
    ap.add_argument("--zig-check", nargs=2, metavar=("CORPUS.json", "RESULTS.json"), help="compare parity.zig's results with the corpus and exit")
    args = ap.parse_args()
    if args.zig_check:
        sys.exit(check_zig_results(*args.zig_check))

    enc_path = Path(args.encoding)
    enc_dir = enc_path if enc_path.is_dir() else enc_path.parent
    enc = load_encoding(args.encoding)
    runner = Runner(enc, Path(args.template))
    if args.zig_corpus:
        write_zig_corpus(runner, enc, enc_dir, args.zig_corpus, args.fuzz, args.seed)
        return

    rows = []
    for label, ok, detail in official(enc, runner, enc_dir):
        rows.append((label, "PASS" if ok else "FAIL", detail))
    for c in cases():
        status, detail = run_case(runner, c)
        rows.append((c["name"], status, detail))

    fuzz_fail = fuzz_skip = 0
    if args.fuzz:
        rng = random.Random(args.seed)
        shown = 0
        for k in range(args.fuzz):
            c = rcase(rng)
            c["name"] = f"fuzz #{k}"
            try:
                runner.reference(c["messages"], c["tools"], c["mode"], c["effort"])
            except Exception:  # noqa: BLE001 - the reference refuses this input, nothing to compare
                fuzz_skip += 1
                continue
            status, detail = run_case(runner, c)
            if status != "PASS":
                fuzz_fail += 1
                if shown < 5:
                    shown += 1
                    print(f"FUZZ FAIL seed={args.seed} #{k}: {detail}")
                    print("  case:", json.dumps(c, ensure_ascii=False)[:1500])

    width = max(len(r[0]) for r in rows)
    fails = gaps = 0
    for name, status, detail in rows:
        if status == "FAIL":
            fails += 1
        if status == "GAP":
            gaps += 1
        if status != "PASS" or args.verbose:
            print(f"{status:4}  {name:<{width}}  {detail}")
    passed = sum(1 for r in rows if r[1] == "PASS")
    print(f"\n{passed} passed, {fails} failed, {gaps} known gaps, out of {len(rows)} named cases")
    if args.fuzz:
        print(f"fuzz: {args.fuzz - fuzz_skip - fuzz_fail} byte-exact, {fuzz_fail} different, {fuzz_skip} refused by the reference (seed {args.seed})")
    sys.exit(1 if fails or fuzz_fail else 0)


if __name__ == "__main__":
    main()
