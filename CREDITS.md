# Credits

This is a trimmed fork of [tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks)
(MIT, Tech2wild), keeping only its TensorFold TP2 lane (two DGX Sparks) and adding four engine patches. The full credits of
that repository, covering every lane and every report it builds on, are in its
[CREDITS.md](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks/blob/ab2d381/CREDITS.md). The
entries below are the ones this lane depends on, in that file's words where they apply.

## Third-party code in this repository

| files | from | license |
|---|---|---|
| `patches/0001-*.patch` | bertholomus's commit `5940208` (the rank-cache writer and token-map builder), author line kept | Apache-2.0 ([third_party/TensorFold/](third_party/TensorFold/)) |
| `patches/0002-*.patch`, `patches/0003-*.patch` | the upstream repository's TP2 lane (opt-in random seed and top_k handling; rank-cache tools on one or two ranks), Tech2wild | Apache-2.0 (changes to TensorFold) |
| `patches/0004-*.patch` to `0007-*.patch` | this fork; `0005` hand-ports [ashhart/TensorFold](https://github.com/ashhart/TensorFold) commits `d7b134cd` and `c4d07a54` by BlivionIaG and `77641ba4` by ashhart | Apache-2.0 (changes to TensorFold) |
| `launch/serve_tp2.sh`, `launch/node_run.sh` | the upstream repository's TP2 lane, adapted from the scripts of [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) and the run commands of [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | Apache-2.0 |
| `launch/build.sh`, `launch/run_build.sh` | the upstream repository's TP2 lane | MIT ([LICENSE](LICENSE)) |
| `launch/chat_template.jinja` | the upstream repository's TP4 lane, written against DeepSeek-V4.1-Flash-1017's `encoding/encoding.py` | MIT |

The TensorFold engine, the recipe's kernel `kit/` and the weights are not included; the steps in the README fetch them
from their sources, whose LICENSE and NOTICE files apply.

## The engine

- **TensorFold** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0): the Zig inference engine.
  Its DeepSeek-V4.1 work on 2 and 4 GB10s is discussed in [ashhart/TensorFold#559](https://github.com/ashhart/TensorFold/issues/559).
  The chunked prefill in `0005` is a port of its commits `d7b134cd` and `c4d07a54` by **BlivionIaG** and `77641ba4` by **ashhart**.
- **bertholomus:** the engine branch `deepseek-v41-zig` of [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold)
  that this lane builds (the dsv41 family, the follower rank, the vision tower), which carries `b8ae0c7` (Lucas Peters),
  `6a1af41` (RDMA GID index and MTU negotiation, sayyidfareed) and `9a0c014` (whole-run expert loader, lepetitprince716-prog);
  the TP2 recipe [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)
  (Apache-2.0, tag `v0.6.1`): the two-node run commands, the kernel kit and its measured settings (`--context 1048576`,
  4 streams, `TF_DS_REPLAY_FLOOR=1`); and the kit mirror on Hugging Face.
- **Tech2wild (tonyd2wild):** the TP2 lane this fork keeps: its launch scripts, the patches `0002`/`0003`, the 1017 chat
  template and the measurements in the upstream repository.

## The model and the weights

- **DeepSeek-AI:** [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and its
  [1017 update](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT), including the Engram tables, the vision
  encoder and the DSpark draft layers.
- **LibertAIDAI:** [DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  (MIT), the pack this lane serves; its card says the tensor names, dtypes and per-tensor bitrates follow Mia-AiLab's
  2.9 bpw release.
- **Mia-AiLab:** [DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) (MIT),
  the 2.9 bpw layout and K map the LibertAI pack follows and bertholomus's kit was built for.
- **turboderp:** [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT), the EXL3 format and quantizer.

If we missed you, or described your work wrong, please open an issue and we will fix it.
