## boot1017x (2026-10-18T02:32:23Z)

EXL3 full, full bench

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 35.98 | 37.78 | 0.194 |
| C2 | 57.65 | 31.14 | 0.262 |
| C3 | 75.97 | 27.72 | 0.308 |
| C4 | 90.48 | 24.83 | 0.349 |
| C5 | 101.98 | 22.85 | 0.396 |
| C6 | 115.76 | 21.33 | 0.418 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 42.31 | 35.47 | 32.97 | 28.49 | 26.65 | 25.8 |
| json | 25.0 | 19.7 | 18.02 | 16.57 | 15.68 | 14.62 |
| narrative | 27.58 | 21.35 | 19.59 | 16.16 | 14.95 | 13.3 |
| prose | 29.13 | 21.25 | 21.6 | 16.82 | 15.74 | 15.25 |
| math | 50.67 | 44.69 | 40.47 | 33.78 | 34.73 | 28.93 |
| reasoning | 35.8 | 29.41 | 26.97 | 27.25 | 22.25 | 20.55 |
| summary | 46.68 | 37.12 | 33.79 | 29.5 | 26.02 | 24.59 |
| format | 45.06 | 40.17 | 28.32 | 30.05 | 26.81 | 27.6 |
| ceiling_count | 41.14 | 36.37 | 33.15 | 31.35 | 29.85 | 27.73 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.677 | 1758.9 |
| 8000 | 11592 | 0.188 | 61730.3 |
| 32000 | 46810 | 0.259 | 181060.9 |
| 64000 | 93335 | 61.709 | 1512.5 |
