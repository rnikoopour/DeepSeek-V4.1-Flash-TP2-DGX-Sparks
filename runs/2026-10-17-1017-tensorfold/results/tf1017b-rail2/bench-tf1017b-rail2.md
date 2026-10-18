## tf1017b-rail2 (2026-10-18T03:45:22Z)

TensorFold TP4, LibertAI 3.5 bpw, BOTH RoCE rails (rocep1s0f0 + roceP2p1s0f0), fleet endpoint :8000 (live, agents may also be calling), completions via DeepSeek encoder

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 93.57 | 107.09 | 0.194 |
| C2 | 136.85 | 81.26 | 0.297 |
| C3 | 167.33 | 67.71 | 0.377 |
| C4 | 189.31 | 58.47 | 0.495 |
| C5 | 202.12 | 50.3 | 0.579 |
| C6 | 213.81 | 44.6 | 0.663 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 143.53 | 100.17 | 89.74 | 70.8 | 67.71 | 59.05 |
| json | 113.51 | 74.5 | 60.82 | 53.86 | 47.75 | 42.38 |
| narrative | 70.75 | 55.54 | 45.88 | 40.75 | 35.27 | 33.43 |
| prose | 80.86 | 58.92 | 52.01 | 48.21 | 41.69 | 37.48 |
| math | 127.03 | 109.43 | 81.41 | 73.05 | 62.89 | 53.29 |
| reasoning | 103.22 | 77.18 | 68.13 | 58.2 | 51.53 | 40.13 |
| summary | 73.48 | 58.74 | 48.86 | 42.64 | 34.13 | 32.24 |
| format | 144.37 | 115.57 | 94.85 | 80.22 | 61.44 | 58.8 |
| ceiling_count | 170.43 | 137.09 | 116.6 | 103.14 | 91.36 | 76.57 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.862 | 1584.6 |
| 8000 | 11592 | 5.418 | 2139.6 |
| 32000 | 46810 | 21.458 | 2181.4 |
| 64000 | 93335 | 42.14 | 2214.9 |
