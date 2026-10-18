## boot1017x (2026-10-18T02:32:23Z to 2026-10-18T02:39:28Z)

EXL3 full, full bench

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 42.3 | 35.5 | 33.0 | 28.5 | 26.7 | 25.8 |
| json | 25.0 | 19.7 | 18.0 | 16.6 | 15.7 | 14.6 |
| math | 50.7 | 44.7 | 40.5 | 33.8 | 34.7 | 28.9 |
| reasoning | 35.8 | 29.4 | 27.0 | 27.3 | 22.3 | 20.5 |
| format | 45.1 | 40.2 | 28.3 | 30.0 | 26.8 | 27.6 |
| summary | 46.7 | 37.1 | 33.8 | 29.5 | 26.0 | 24.6 |
| prose | 29.1 | 21.3 | 21.6 | 16.8 | 15.7 | 15.2 |
| narrative | 27.6 | 21.3 | 19.6 | 16.2 | 14.9 | 13.3 |
| counting (ceiling) | 41.1 | 36.4 | 33.1 | 31.4 | 29.8 | 27.7 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 41.4 | 65.8 | 91.2 | 105.4 | 121.0 | 144.7 |
| json | 24.1 | 37.2 | 50.7 | 61.8 | 73.4 | 82.2 |
| math | 48.5 | 85.3 | 115.4 | 120.8 | 163.1 | 164.5 |
| reasoning | 35.0 | 56.3 | 77.1 | 104.3 | 103.8 | 115.8 |
| format | 42.0 | 73.8 | 73.1 | 107.5 | 110.4 | 137.7 |
| summary | 41.5 | 62.0 | 82.8 | 98.8 | 99.5 | 119.8 |
| prose | 28.4 | 40.5 | 61.8 | 64.4 | 74.9 | 87.3 |
| narrative | 27.0 | 40.2 | 55.7 | 60.8 | 69.7 | 74.2 |
| counting (ceiling) | 40.2 | 71.0 | 97.3 | 116.2 | 136.3 | 152.4 |

### TTFT: mean time to first token (s)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 0.13 | 0.22 | 0.22 | 0.28 | 0.27 | 0.27 |
| json | 0.16 | 0.26 | 0.33 | 0.40 | 0.41 | 0.43 |
| math | 0.20 | 0.24 | 0.26 | 0.29 | 0.33 | 0.35 |
| reasoning | 0.16 | 0.26 | 0.32 | 0.29 | 0.40 | 0.43 |
| format | 0.21 | 0.26 | 0.30 | 0.33 | 0.36 | 0.38 |
| summary | 0.36 | 0.47 | 0.62 | 0.74 | 0.86 | 0.98 |
| prose | 0.16 | 0.19 | 0.21 | 0.23 | 0.26 | 0.24 |
| narrative | 0.16 | 0.20 | 0.21 | 0.25 | 0.28 | 0.27 |
| counting (ceiling) | 0.16 | 0.19 | 0.18 | 0.22 | 0.22 | 0.23 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 41.4 | 34.3 | 32.0 | 27.5 | 25.8 | 25.0 |
| json | 24.2 | 18.8 | 17.0 | 15.5 | 14.8 | 13.8 |
| math | 48.5 | 42.7 | 38.7 | 32.3 | 33.0 | 27.7 |
| reasoning | 35.0 | 28.5 | 26.0 | 26.4 | 21.4 | 19.8 |
| format | 42.1 | 37.3 | 26.5 | 27.9 | 25.0 | 25.6 |
| summary | 41.5 | 32.9 | 29.3 | 25.6 | 22.1 | 20.9 |
| prose | 28.4 | 20.8 | 21.0 | 16.5 | 15.4 | 14.9 |
| narrative | 27.0 | 20.9 | 19.2 | 15.8 | 14.6 | 13.1 |
| counting (ceiling) | 40.2 | 35.5 | 32.5 | 30.6 | 29.2 | 27.1 |

### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 1.68 | 1,759 |
| 8K | 11,592 | 0.19 | 61,730 |
| 32K | 46,810 | 0.26 | 181,061 |
| 64K | 93,335 | 61.71 | 1,512 |
