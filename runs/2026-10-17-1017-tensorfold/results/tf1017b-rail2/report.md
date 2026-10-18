## tf1017b-rail2 (2026-10-18T03:45:22Z to 2026-10-18T03:49:15Z)

TensorFold TP4, LibertAI 3.5 bpw, BOTH RoCE rails (rocep1s0f0 + roceP2p1s0f0), fleet endpoint :8000 (live, agents may also be calling), completions via DeepSeek encoder

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 143.5 | 100.2 | 89.7 | 70.8 | 67.7 | 59.0 |
| json | 113.5 | 74.5 | 60.8 | 53.9 | 47.8 | 42.4 |
| math | 127.0 | 109.4 | 81.4 | 73.0 | 62.9 | 53.3 |
| reasoning | 103.2 | 77.2 | 68.1 | 58.2 | 51.5 | 40.1 |
| format | 144.4 | 115.6 | 94.9 | 80.2 | 61.4 | 58.8 |
| summary | 73.5 | 58.7 | 48.9 | 42.6 | 34.1 | 32.2 |
| prose | 80.9 | 58.9 | 52.0 | 48.2 | 41.7 | 37.5 |
| narrative | 70.7 | 55.5 | 45.9 | 40.8 | 35.3 | 33.4 |
| counting (ceiling) | 170.4 | 137.1 | 116.6 | 103.1 | 91.4 | 76.6 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 132.3 | 179.9 | 238.8 | 250.0 | 293.5 | 303.9 |
| json | 89.9 | 114.8 | 137.8 | 162.0 | 178.4 | 180.7 |
| math | 111.2 | 190.5 | 211.0 | 250.0 | 262.1 | 267.6 |
| reasoning | 94.7 | 139.3 | 180.3 | 199.9 | 220.8 | 209.3 |
| format | 115.5 | 168.1 | 204.1 | 223.6 | 210.3 | 240.8 |
| summary | 61.1 | 93.2 | 105.9 | 116.6 | 114.3 | 138.5 |
| prose | 76.2 | 108.1 | 139.2 | 170.0 | 181.8 | 194.5 |
| narrative | 67.8 | 100.9 | 121.6 | 142.5 | 155.6 | 175.2 |
| counting (ceiling) | 160.1 | 252.2 | 320.4 | 374.6 | 412.0 | 414.4 |

### TTFT: mean time to first token (s)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 0.12 | 0.20 | 0.26 | 0.32 | 0.39 | 0.46 |
| json | 0.20 | 0.32 | 0.39 | 0.45 | 0.55 | 0.64 |
| math | 0.23 | 0.28 | 0.37 | 0.46 | 0.56 | 0.65 |
| reasoning | 0.18 | 0.28 | 0.35 | 0.45 | 0.53 | 0.62 |
| format | 0.22 | 0.33 | 0.42 | 0.54 | 0.66 | 0.75 |
| summary | 0.37 | 0.63 | 0.75 | 1.15 | 1.23 | 1.36 |
| prose | 0.11 | 0.17 | 0.23 | 0.29 | 0.35 | 0.41 |
| narrative | 0.11 | 0.18 | 0.24 | 0.30 | 0.37 | 0.43 |
| counting (ceiling) | 0.10 | 0.14 | 0.19 | 0.24 | 0.29 | 0.33 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 132.3 | 91.4 | 80.6 | 63.7 | 59.8 | 52.0 |
| json | 89.9 | 58.0 | 47.4 | 41.4 | 36.0 | 31.7 |
| math | 111.2 | 95.3 | 70.8 | 62.5 | 53.4 | 45.3 |
| reasoning | 94.7 | 69.9 | 61.0 | 51.8 | 45.3 | 35.8 |
| format | 115.5 | 84.1 | 68.1 | 55.9 | 44.3 | 40.5 |
| summary | 61.1 | 47.0 | 37.7 | 30.5 | 24.8 | 23.9 |
| prose | 76.2 | 55.1 | 47.8 | 43.8 | 37.6 | 33.5 |
| narrative | 67.9 | 52.4 | 43.0 | 37.8 | 32.7 | 30.7 |
| counting (ceiling) | 160.2 | 127.3 | 106.9 | 93.7 | 82.5 | 69.3 |

### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 1.86 | 1,585 |
| 8K | 11,592 | 5.42 | 2,140 |
| 32K | 46,810 | 21.46 | 2,181 |
| 64K | 93,335 | 42.14 | 2,215 |
