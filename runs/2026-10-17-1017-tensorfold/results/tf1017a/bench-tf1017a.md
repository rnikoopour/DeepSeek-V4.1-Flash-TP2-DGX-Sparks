## tf1017a (2026-10-18T03:01:50Z)

TensorFold TP4, LibertAI 3.5 bpw full EXL3, rail 1, completions via DeepSeek encoder

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 89.78 | 101.79 | 0.191 |
| C2 | 132.78 | 78.04 | 0.285 |
| C3 | 158.99 | 64.01 | 0.382 |
| C4 | 180.3 | 55.25 | 0.478 |
| C5 | 192.94 | 47.7 | 0.572 |
| C6 | 202.9 | 42.05 | 0.674 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 134.76 | 97.12 | 84.65 | 65.97 | 63.51 | 55.33 |
| json | 107.61 | 71.72 | 59.5 | 50.81 | 45.07 | 39.66 |
| narrative | 67.28 | 52.56 | 42.95 | 38.62 | 33.46 | 31.95 |
| prose | 78.04 | 56.42 | 49.11 | 45.86 | 39.05 | 35.6 |
| math | 119.86 | 104.6 | 76.81 | 68.36 | 58.99 | 50.09 |
| reasoning | 97.71 | 73.45 | 63.79 | 54.85 | 49.8 | 37.88 |
| summary | 69.73 | 56.41 | 46.05 | 41.57 | 32.89 | 30.54 |
| format | 139.36 | 112.02 | 89.2 | 75.97 | 58.82 | 55.35 |
| ceiling_count | 165.24 | 129.19 | 109.35 | 96.14 | 85.02 | 71.53 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.829 | 1613.2 |
| 8000 | 11592 | 6.19 | 1872.6 |
| 32000 | 46810 | 24.717 | 1893.8 |
| 64000 | 93335 | 48.395 | 1928.6 |
