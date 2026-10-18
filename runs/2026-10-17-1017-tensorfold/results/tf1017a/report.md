## tf1017a (2026-10-18T03:01:50Z to 2026-10-18T03:06:01Z)

TensorFold TP4, LibertAI 3.5 bpw full EXL3, rail 1, completions via DeepSeek encoder

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 134.8 | 97.1 | 84.7 | 66.0 | 63.5 | 55.3 |
| json | 107.6 | 71.7 | 59.5 | 50.8 | 45.1 | 39.7 |
| math | 119.9 | 104.6 | 76.8 | 68.4 | 59.0 | 50.1 |
| reasoning | 97.7 | 73.4 | 63.8 | 54.9 | 49.8 | 37.9 |
| format | 139.4 | 112.0 | 89.2 | 76.0 | 58.8 | 55.4 |
| summary | 69.7 | 56.4 | 46.0 | 41.6 | 32.9 | 30.5 |
| prose | 78.0 | 56.4 | 49.1 | 45.9 | 39.1 | 35.6 |
| narrative | 67.3 | 52.6 | 42.9 | 38.6 | 33.5 | 31.9 |
| counting (ceiling) | 165.2 | 129.2 | 109.4 | 96.1 | 85.0 | 71.5 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 124.3 | 174.2 | 225.0 | 233.3 | 275.6 | 284.6 |
| json | 87.5 | 114.7 | 136.1 | 153.2 | 168.6 | 171.1 |
| math | 108.1 | 182.0 | 200.0 | 235.4 | 247.8 | 252.2 |
| reasoning | 90.0 | 133.7 | 169.4 | 188.0 | 214.3 | 197.8 |
| format | 112.2 | 165.9 | 192.7 | 213.2 | 203.9 | 228.4 |
| summary | 58.4 | 92.6 | 103.1 | 123.1 | 115.3 | 136.9 |
| prose | 73.3 | 103.5 | 131.6 | 161.2 | 169.9 | 184.8 |
| narrative | 64.4 | 95.7 | 114.1 | 135.1 | 148.0 | 167.3 |
| counting (ceiling) | 155.0 | 237.0 | 301.4 | 349.6 | 384.1 | 387.4 |

### TTFT: mean time to first token (s)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 0.13 | 0.21 | 0.28 | 0.34 | 0.42 | 0.49 |
| json | 0.19 | 0.28 | 0.38 | 0.47 | 0.57 | 0.66 |
| math | 0.19 | 0.29 | 0.38 | 0.47 | 0.57 | 0.68 |
| reasoning | 0.19 | 0.27 | 0.36 | 0.45 | 0.53 | 0.65 |
| format | 0.22 | 0.32 | 0.44 | 0.56 | 0.65 | 0.77 |
| summary | 0.37 | 0.54 | 0.72 | 0.90 | 1.08 | 1.26 |
| prose | 0.12 | 0.18 | 0.24 | 0.31 | 0.37 | 0.43 |
| narrative | 0.12 | 0.19 | 0.25 | 0.31 | 0.38 | 0.44 |
| counting (ceiling) | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 124.3 | 88.5 | 75.9 | 59.4 | 56.1 | 48.7 |
| json | 87.6 | 58.0 | 46.9 | 39.2 | 34.0 | 29.9 |
| math | 108.2 | 91.0 | 67.1 | 58.9 | 50.5 | 42.7 |
| reasoning | 90.0 | 67.1 | 57.3 | 48.7 | 44.0 | 33.8 |
| format | 112.3 | 83.0 | 64.3 | 53.4 | 43.0 | 38.4 |
| summary | 58.4 | 46.8 | 36.8 | 32.4 | 25.2 | 23.7 |
| prose | 73.3 | 52.7 | 45.2 | 41.5 | 35.1 | 31.9 |
| narrative | 64.4 | 49.7 | 40.3 | 35.8 | 31.1 | 29.3 |
| counting (ceiling) | 155.0 | 119.7 | 100.5 | 87.4 | 76.9 | 64.8 |

### Cold prefill (unique prompt, 1-token reply; TTFT = whole prefill)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 2,950 | 1.83 | 1,613 |
| 8K | 11,592 | 6.19 | 1,873 |
| 32K | 46,810 | 24.72 | 1,894 |
| 64K | 93,335 | 48.40 | 1,929 |
