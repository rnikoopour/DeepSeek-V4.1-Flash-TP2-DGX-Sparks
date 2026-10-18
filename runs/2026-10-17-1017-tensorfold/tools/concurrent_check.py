#!/usr/bin/env python3
"""concurrent_check.py BASE: greedy (temperature 0) replies to 8 different chat prompts, once one at a time and once all at the
same time (8 streams), must be identical text and token count. A light stand-in for the recipe's concurrent == solo gate
(the recipe's long_check.py does it with long prompts, which would stall live traffic)."""
import concurrent.futures as cf, json, sys, urllib.request
base = sys.argv[1].rstrip("/")
PROMPTS = ["What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
           "Explain the difference between a mutex and a semaphore with a short C example.",
           "Write a haiku about autumn rain on a tin roof.",
           "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?",
           "Translate into French and German: 'The library opens at eight, but the archive needs an appointment.'",
           "List five practical tips for reducing memory fragmentation in a long-running C++ server.",
           "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
           "Summarize the plot of Romeo and Juliet in three sentences."]
def ask(p):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
solo = [ask(p) for p in PROMPTS]
with cf.ThreadPoolExecutor(len(PROMPTS)) as ex: conc = list(ex.map(ask, PROMPTS))
eq = [a == b for a, b in zip(solo, conc)]
for i, (a, b) in enumerate(zip(solo, conc)): print(("PASS" if a == b else "FAIL"), "prompt %d" % i, a[1], b[1])
print("RESULT concurrent == solo: %d of %d identical" % (sum(eq), len(eq)))
sys.exit(0 if all(eq) else 1)
