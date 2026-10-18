## tf1017-stock-c1c32 (2026-10-18T16:15:22Z)

STOCK (before the Keys overlay swap). TensorFold TP4, LibertAI EXL3 3.5 bpw, both RoCE lanes, patch 0004, fleet endpoint :8000 live (agents may also be calling), completions via DeepSeek encoder. Server runs 16 streams; C32 queues.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 98.55 | 111.98 | 0.176 |
| C2 | 146.46 | 86.36 | 0.265 |
| C4 | 198.01 | 61.13 | 0.449 |
| C8 | 221.66 | 34.64 | 0.807 |
| C12 | 250.9 | 26.32 | 1.176 |
| C16 | 270.76 | 21.66 | 1.539 |
| C32 | 260.52 | 19.05 | 5.96 |

### Per-stream tok/s by category

| category | C1 | C2 | C4 | C8 | C12 | C16 | C32 |
|---|---|---|---|---|---|---|---|
| coding | 148.74 | 109.9 | 73.57 | 41.65 | 35.69 | 27.93 | 25.65 |
| json | 117.78 | 79.99 | 56.59 | 34.61 | 25.87 | 21.86 | 17.77 |
| narrative | 73.93 | 58.56 | 42.77 | 29.06 | 18.39 | 16.57 | 14.8 |
| prose | 85.89 | 61.73 | 50.63 | 27.19 | 22.49 | 18.72 | 17.44 |
| math | 132.79 | 114.61 | 75.7 | 44.24 | 33.97 | 26.93 | 24.62 |
| reasoning | 108.37 | 81.09 | 60.53 | 31.37 | 26.56 | 22.98 | 19.7 |
| summary | 76.81 | 62.18 | 46.58 | 23.84 | 17.9 | 15.24 | 12.72 |
| format | 151.57 | 122.79 | 82.64 | 45.16 | 29.66 | 23.08 | 19.68 |
| ceiling_count | 179.71 | 142.33 | 107.18 | 67.58 | 40.12 | 31.61 | 29.67 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.376 | 2143.7 |
| 8000 | 11592 | 5.115 | 2266.1 |
| 32000 | 46810 | 20.685 | 2263.0 |
| 64000 | 93335 | 42.18 | 2212.8 |
