## tf1017a-c8c16 (2026-10-18T03:21:31Z to 2026-10-18T03:23:52Z)

TensorFold TP4 rail 1, C8 and C16

Prompt set `v1`, temperature 0, thinking off, streaming; one batch per cell (C streams released together). Short prompts (about 30-120 tokens), 150-256 token budgets.

### Decode: per-stream tok/s after the first token

| category | C8 | C16 |
|---|---|---|
| coding | 46.7 | 25.3 |
| json | 32.1 | 20.0 |
| math | 40.8 | 24.5 |
| reasoning | 28.7 | 20.5 |
| format | 41.7 | 20.4 |
| summary | 21.5 | 13.9 |
| prose | 25.5 | 17.1 |
| narrative | 27.1 | 15.4 |
| counting (ceiling) | 62.0 | 28.0 |

### Aggregate throughput: tok/s across all streams (wall time, TTFT included)

| category | C8 | C16 |
|---|---|---|
| coding | 320.8 | 346.2 |
| json | 184.7 | 215.6 |
| math | 274.2 | 322.8 |
| reasoning | 200.9 | 275.6 |
| format | 227.7 | 227.7 |
| summary | 130.7 | 157.5 |
| prose | 178.1 | 226.6 |
| narrative | 184.8 | 207.2 |
| counting (ceiling) | 445.3 | 402.2 |

### TTFT: mean time to first token (s)

| category | C8 | C16 |
|---|---|---|
| coding | 0.61 | 1.15 |
| json | 0.83 | 1.59 |
| math | 0.84 | 1.64 |
| reasoning | 0.81 | 1.58 |
| format | 0.98 | 1.90 |
| summary | 1.60 | 3.05 |
| prose | 0.55 | 1.05 |
| narrative | 0.57 | 1.06 |
| counting (ceiling) | 0.44 | 0.87 |

### End-to-end per stream: tok/s over the whole request, TTFT included

| category | C8 | C16 |
|---|---|---|
| coding | 40.8 | 22.0 |
| json | 23.9 | 14.0 |
| math | 34.7 | 20.3 |
| reasoning | 25.7 | 17.6 |
| format | 29.6 | 14.7 |
| summary | 17.1 | 10.3 |
| prose | 23.1 | 15.0 |
| narrative | 24.5 | 13.9 |
| counting (ceiling) | 55.7 | 25.4 |

