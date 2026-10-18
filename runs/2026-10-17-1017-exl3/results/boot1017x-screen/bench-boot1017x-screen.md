## boot1017x-screen (2026-10-18T02:26:36Z)

EXL3 full, screen

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 32.62 | 34.18 | 0.214 |
| C3 | 72.34 | 27.66 | 0.684 |
| C6 | 101.98 | 18.85 | 0.509 |

### Per-stream tok/s by category

| category | C1 | C3 | C6 |
|---|---|---|---|
| coding | 41.55 | 31.06 | 24.7 |
| json | 23.8 | 18.29 | 13.82 |
| narrative | 25.5 | 18.2 | 12.65 |
| prose | 23.7 | 20.66 | 14.65 |
| math | 44.61 | 40.01 | 29.22 |
| reasoning | 32.96 | 29.39 | 22.65 |
| summary | 41.61 | 31.33 | 24.67 |
| format | 39.71 | 32.34 | 8.44 |
| ceiling_count | 35.97 | 33.29 | 21.76 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 8000 | 11592 | 9.066 | 1278.6 |
| 32000 | 46810 | 25.693 | 1821.9 |
