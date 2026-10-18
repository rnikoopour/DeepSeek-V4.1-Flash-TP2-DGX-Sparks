## boot1017x-screen (2026-10-18T02:26:36Z to 2026-10-18T02:30:34Z)

EXL3 full, screen

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 41.6 | 31.1 | 24.7 |
| json | 23.8 | 18.3 | 13.8 |
| math | 44.6 | 40.0 | 29.2 |
| reasoning | 33.0 | 29.4 | 22.6 |
| format | 39.7 | 32.3 | 8.4 |
| summary | 41.6 | 31.3 | 24.7 |
| prose | 23.7 | 20.7 | 14.7 |
| narrative | 25.5 | 18.2 | 12.6 |
| counting (ceiling) | 36.0 | 33.3 | 21.8 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 40.6 | 86.1 | 138.3 |
| json | 23.0 | 51.2 | 77.2 |
| math | 42.7 | 114.2 | 164.6 |
| reasoning | 32.0 | 84.6 | 128.2 |
| format | 37.2 | 81.4 | 47.6 |
| summary | 37.5 | 49.0 | 107.9 |
| prose | 23.1 | 59.6 | 82.2 |
| narrative | 24.9 | 52.6 | 69.8 |
| counting (ceiling) | 35.3 | 93.9 | 118.7 |

### TTFT: mean time to first token (s)

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 0.14 | 0.24 | 0.31 |
| json | 0.17 | 0.34 | 0.49 |
| math | 0.22 | 0.28 | 0.37 |
| reasoning | 0.22 | 0.26 | 0.32 |
| format | 0.23 | 0.29 | 0.39 |
| summary | 0.39 | 3.62 | 1.65 |
| prose | 0.18 | 0.22 | 0.25 |
| narrative | 0.17 | 0.22 | 0.28 |
| counting (ceiling) | 0.15 | 0.20 | 0.34 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 40.6 | 30.1 | 23.9 |
| json | 23.0 | 17.2 | 12.9 |
| math | 42.7 | 38.1 | 27.8 |
| reasoning | 32.0 | 28.5 | 22.0 |
| format | 37.2 | 30.2 | 8.3 |
| summary | 37.5 | 16.9 | 18.9 |
| prose | 23.1 | 20.1 | 14.4 |
| narrative | 24.9 | 17.8 | 12.4 |
| counting (ceiling) | 35.3 | 32.5 | 21.2 |

### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8K | 11,592 | 9.07 | 1,279 |
| 32K | 46,810 | 25.69 | 1,822 |
