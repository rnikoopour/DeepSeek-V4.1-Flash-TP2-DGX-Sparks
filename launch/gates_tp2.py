#!/usr/bin/env python3
"""The P0-1 exact-reply gates for the TP2 lane: drafted == serial, concurrent == solo, images 9/9 and 18/18.

Every gate compares the server's own reply fingerprint, ``tensorfold.token_sha`` (sha256 of the reply's token ids, first
12 hex digits), so a mismatch is a token-level difference, not a formatting one. Greedy (temperature 0), thinking off,
so each reply is a pure function of the prompt and the engine.

  python3 gates_tp2.py text  --out F            # 12 text prompts, one at a time, drafting on (the drafted pass)
  python3 gates_tp2.py text  --out F --expect-drafts false   # the serial pass, engine restarted with --no-drafts
  python3 gates_tp2.py same  A.json B.json      # drafted == serial
  python3 gates_tp2.py text-parallel --ref F    # concurrent == solo on the same running engine
  python3 gates_tp2.py images --out F           # 9 image prompts, one at a time
  python3 gates_tp2.py images-parallel --ref F --parallel 4   # 18 image requests, 4 at a time
  python3 gates_tp2.py fair                     # a short prompt sent 2 s into a long cold pass

BASE defaults to http://127.0.0.1:8000; MODEL to deepseek-v4.1-flash. Exit status 1 on any failed gate.
Standard library only (the nodes' host Python).
"""

import argparse
import base64
import concurrent.futures as cf
import json
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

TEXT_PROMPTS = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    "Explain the difference between a mutex and a semaphore with a short C example.",
    "Write a haiku about autumn rain on a tin roof.",
    "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?",
    "Translate into French and German: 'The library opens at eight, but the archive needs an appointment.'",
    "List five practical tips for reducing memory fragmentation in a long-running C++ server.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "In one paragraph, explain why a hash table lookup is O(1) on average and when it stops being O(1).",
    "Convert 27 degrees Celsius to Fahrenheit and show the arithmetic.",
    "Write a SQL query: the five customers with the highest total order value in 2025.",
    "What does this regular expression match: ^(?:[a-z0-9_-]+\\.)+[a-z]{2,}$ ?",
]

IMAGE_PROMPT = "What colour is this image? Answer with one word."
IMAGE_COLOURS = [
    (220, 30, 30), (30, 160, 60), (40, 70, 210), (230, 200, 40), (40, 190, 200),
    (200, 60, 190), (230, 130, 30), (120, 60, 190), (110, 110, 110),
]


def png(width, height, colour, mark):
    """A width x height RGB PNG of one colour with a white square at mark (a corner: 0-3)."""
    px = [colour] * width
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            white = False
            if mark in (0, 1) and y < height // 3:
                white = (x < width // 3) if mark == 0 else (x >= width - width // 3)
            if mark in (2, 3) and y >= height - height // 3:
                white = (x < width // 3) if mark == 2 else (x >= width - width // 3)
            row += bytes((255, 255, 255)) if white else bytes(px[x])
        rows.append(bytes(row))

    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def image_url(i):
    data = base64.b64encode(png(64, 64, IMAGE_COLOURS[i], i % 4)).decode()
    return "data:image/png;base64," + data


class Lane:
    def __init__(self, base, model):
        self.base = base.rstrip("/")
        self.model = model

    def ask(self, prompt, max_tokens, image=None, timeout=900):
        content = prompt if image is None else [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url(image)}},
        ]
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        tf = out.get("tensorfold") or {}
        return {
            "sha": tf.get("token_sha"),
            "drafts": tf.get("drafts"),
            "tokens": out["usage"]["completion_tokens"],
            "prompt_tokens": out["usage"]["prompt_tokens"],
            "ttft": tf.get("time_to_first_token"),
            "text": (out["choices"][0]["message"].get("content") or ""),
            "seconds": round(time.time() - t, 2),
        }


def run_all(lane, prompts, max_tokens, images=None, together=1):
    """Solo (together=1) or `together` at a time; returns one record per request in prompt order."""
    def one(i):
        return lane.ask(prompts[i], max_tokens, None if images is None else images[i])

    if together <= 1:
        return [one(i) for i in range(len(prompts))]
    with cf.ThreadPoolExecutor(together) as ex:
        return list(ex.map(one, range(len(prompts))))


def save(path, kind, prompts, recs):
    with open(path, "w") as f:
        json.dump({"kind": kind, "prompts": prompts, "replies": recs}, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)


def compare(a, b, label):
    ok = 0
    for i, (x, y) in enumerate(zip(a, b)):
        same = x["sha"] == y["sha"]
        ok += same
        print(("PASS" if same else "FAIL") + f" {label} {i}: {x['sha']} vs {y['sha']}"
              + ("" if same else f"  ({x['text'][:60]!r} vs {y['text'][:60]!r})"))
    print(f"RESULT {label}: {ok} of {len(a)} identical")
    return ok == len(a) and len(a) == len(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="deepseek-v4.1-flash")
    s = p.add_subparsers(dest="cmd", required=True)

    t = s.add_parser("text", help="the 12 text prompts, one at a time")
    t.add_argument("--out", required=True)
    t.add_argument("--expect-drafts", choices=["true", "false"], default=None)
    t.add_argument("--tokens", type=int, default=200)

    tp = s.add_parser("text-parallel", help="concurrent == solo")
    tp.add_argument("--ref", required=True)
    tp.add_argument("--tokens", type=int, default=200)
    tp.add_argument("--together", type=int, default=12)
    tp.add_argument("--out")

    im = s.add_parser("images", help="the 9 image prompts, one at a time")
    im.add_argument("--out", required=True)
    im.add_argument("--tokens", type=int, default=12)

    ip = s.add_parser("images-parallel", help="18 image requests at N a time, against the solo replies")
    ip.add_argument("--ref", required=True)
    ip.add_argument("--parallel", type=int, default=4)
    ip.add_argument("--tokens", type=int, default=12)
    ip.add_argument("--out")

    sm = s.add_parser("same", help="two recorded passes")
    sm.add_argument("a")
    sm.add_argument("b")

    fa = s.add_parser("fair", help="a short prompt sent into a long cold pass: its TTFT must stay small")
    fa.add_argument("--tokens", type=int, default=9000, help="words in the long prompt (~4.4 tokens a word)")
    fa.add_argument("--limit", type=float, default=10.0, help="the short prompt's TTFT must stay under this")
    fa.add_argument("--wait", type=float, default=2.0, help="seconds into the long pass the short one is sent")

    a = p.parse_args()
    lane = Lane(a.base, a.model)

    if a.cmd == "fair":
        import random
        random.seed(a.tokens)
        words = " ".join("f%d%s" % (i, "z" * random.randint(1, 6)) for i in range(a.tokens))
        long_prompt = "Task C: list these items in order, one per line.\n" + words
        res = {}

        def run_long():
            res["long"] = lane.ask(long_prompt, 8)

        th = threading.Thread(target=run_long)
        th.start()
        time.sleep(a.wait)
        short = lane.ask("What is 17*23? nonce %d" % int(time.time()), 32)
        th.join()
        ttft = short.get("ttft") if short.get("ttft") is not None else short["seconds"]
        print(f"RESULT long pass: {res['long']['prompt_tokens']} prompt tokens, {res['long']['seconds']} s")
        print(f"RESULT short prompt {a.wait:.0f} s into it: {short['seconds']} s wall, ttft {short.get('ttft')}")
        ok = ttft < a.limit
        print(("PASS " if ok else "FAIL ") + f"short prompt behind a long pass under {a.limit:.0f} s")
        sys.exit(0 if ok else 1)

    if a.cmd == "same":
        x, y = load(a.a), load(a.b)
        ok = compare(x["replies"], y["replies"], "drafted == serial")
        sys.exit(0 if ok else 1)

    if a.cmd == "text":
        recs = run_all(lane, TEXT_PROMPTS, a.tokens)
        save(a.out, "text", TEXT_PROMPTS, recs)
        if a.expect_drafts:
            want = a.expect_drafts == "true"
            bad = [i for i, r in enumerate(recs) if r["drafts"] is not want]
            print(("PASS" if not bad else "FAIL") + f" every reply drafted={want}" + ("" if not bad else f" (requests {bad})"))
            if bad:
                sys.exit(1)
        print(f"RESULT text pass: {len(recs)} replies recorded in {a.out}")
        sys.exit(0)

    if a.cmd == "text-parallel":
        ref = load(a.ref)
        recs = run_all(lane, TEXT_PROMPTS, a.tokens, together=a.together)
        if a.out:
            save(a.out, "text-parallel", TEXT_PROMPTS, recs)
        sys.exit(0 if compare(ref["replies"], recs, "concurrent == solo") else 1)

    if a.cmd == "images":
        recs = run_all(lane, [IMAGE_PROMPT] * len(IMAGE_COLOURS), a.tokens, images=list(range(len(IMAGE_COLOURS))))
        save(a.out, "images", [IMAGE_PROMPT] * len(IMAGE_COLOURS), recs)
        good = sum(1 for r in recs if r["sha"] and r["text"].strip())
        print(f"RESULT images alone: {good}/{len(recs)}" + ("" if good == len(recs) else " FAILED"))
        for i, r in enumerate(recs):
            print(f"  {i}: {r['text'].strip()[:40]!r} sha={r['sha']}")
        sys.exit(0 if good == len(recs) else 1)

    if a.cmd == "images-parallel":
        ref = load(a.ref)
        n = len(IMAGE_COLOURS)
        order = list(range(n)) + list(range(n))       # 18 requests, each prompt twice
        recs = run_all(lane, [IMAGE_PROMPT] * len(order), a.tokens, images=order, together=a.parallel)
        if a.out:
            save(a.out, "images-parallel", [IMAGE_PROMPT] * len(order), recs)
        ok = sum(1 for i, r in enumerate(recs) if r["sha"] == ref["replies"][order[i]]["sha"])
        for i, r in enumerate(recs):
            same = r["sha"] == ref["replies"][order[i]]["sha"]
            print(("PASS" if same else "FAIL") + f" image {order[i]} at {a.parallel} a time: {r['sha']}"
                  + ("" if same else f" vs solo {ref['replies'][order[i]]['sha']}"))
        print(f"RESULT images {a.parallel} at a time: {ok}/{len(recs)}" + ("" if ok == len(recs) else " FAILED"))
        sys.exit(0 if ok == len(recs) else 1)


if __name__ == "__main__":
    main()
