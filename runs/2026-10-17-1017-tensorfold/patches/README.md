# Engine patches on bertholomus/TensorFold `deepseek-v41-zig-tp4` @ 5940208

Apply in order with `git am patches/*.patch` (author lines are kept).

| patch | what it changes | origin |
|---|---|---|
| `0001-...captured-Triton-variant...` | `aot.zig`: a launch with no captured Triton variant runs the closest more general one instead of stopping on `MissingTritonVariant`; names the variant once on stderr; `TF_AOT_WEAKEST=1` forces the most general variant for gating | commit `47a8910` by bertholomus, public on the fork's `deepseek-v41-zig` branch; cherry-picked unchanged |
| `0002-...vis 16-byte aligned...` | `prompt.zig`: the prompt indexer passes each row block's `vis` 16-byte aligned | commit `b8ae0c7` (PR #4 to the fork) by Lucas Peters, public on the fork's `deepseek-v41-zig` branch; cherry-picked unchanged |
| `0003-...TF_RDMA_GID...` | `rdma.zig`, `model.zig`, `layer_main.zig`: the RDMA rings' source GID index is read from `TF_RDMA_GID` (default 5 unchanged) instead of being fixed at 5 | ours; the approach is the one ecohash-co described in ashhart/TensorFold#559 |

Together 0001 and 0002 cover the `_index_score` gap described by ecohash-co in #559 (prompts of 131,073-149,796 tokens); the last-chunk fallback is exercised and logged on this cluster, see `../logs/long_check1.log`.
Not applied: `TF_DS_PROMPT_SP=rows` (sequence-parallel prompt rows, reported by ecohash-co as +12-17% prefill; its code is not public).
