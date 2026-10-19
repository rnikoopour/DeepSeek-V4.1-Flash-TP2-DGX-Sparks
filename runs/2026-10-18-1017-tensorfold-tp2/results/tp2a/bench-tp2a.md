## tp2a (2026-10-19T00:04:06Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 73.08 | 84.16 | 0.243 |
| C2 | 106.04 | 62.81 | 0.366 |
| C3 | 123.7 | 49.3 | 0.487 |
| C4 | 135.75 | 41.39 | 0.612 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| coding | 105.63 | 81.39 | 64.17 | 51.56 |
| json | 92.12 | 60.76 | 47.31 | 38.78 |
| narrative | 61.6 | 42.71 | 35.43 | 29.93 |
| prose | 55.43 | 49.83 | 43.07 | 39.06 |
| math | 99.55 | 71.99 | 56.92 | 48.15 |
| reasoning | 82.0 | 59.56 | 49.93 | 41.48 |
| summary | 62.87 | 46.29 | 36.52 | 33.79 |
| format | 114.09 | 89.94 | 61.08 | 48.4 |
| ceiling_count | 135.88 | 112.31 | 79.74 | 59.58 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.535 | 1921.6 |
| 8000 | 11592 | 5.317 | 2180.2 |
| 32000 | 46810 | 21.282 | 2199.5 |
| 64000 | 93335 | 43.723 | 2134.7 |
| 128000 | 186773 | 91.359 | 2044.4 |
