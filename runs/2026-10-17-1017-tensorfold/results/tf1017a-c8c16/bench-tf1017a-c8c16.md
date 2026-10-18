## tf1017a-c8c16 (2026-10-18T03:21:31Z)

TensorFold TP4 rail 1, C8 and C16

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C8 | 212.75 | 33.02 | 0.85 |
| C16 | 247.39 | 19.65 | 1.628 |

### Per-stream tok/s by category

| category | C8 | C16 |
|---|---|---|
| coding | 46.73 | 25.26 |
| json | 32.1 | 20.04 |
| narrative | 27.11 | 15.42 |
| prose | 25.5 | 17.11 |
| math | 40.82 | 24.5 |
| reasoning | 28.65 | 20.55 |
| summary | 21.54 | 13.92 |
| format | 41.7 | 20.43 |
| ceiling_count | 62.02 | 28.01 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
