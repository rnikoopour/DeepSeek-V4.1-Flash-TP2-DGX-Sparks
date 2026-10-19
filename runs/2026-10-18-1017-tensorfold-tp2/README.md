# DeepSeek-V4.1-Flash-1017 on TensorFold, TP2 on two DGX Sparks (EXL3 2.9 bpw), 2026-10-18

The 1017 update served by [TensorFold](https://github.com/ashhart/TensorFold)'s native engine on **two** DGX Sparks, with LibertAI's 2.9 bpw EXL3 pack (196.3 GiB, 98.9 GiB of weights per node). This folder holds the engine patches, the scripts with every env var, and the results. **Status: deployed 2026-10-19 00:01 UTC (first boot, no deaths), deployment gates pass; the bench and eval are running** (see [STATUS.md](STATUS.md)).

## What runs

| piece | what | source |
|---|---|---|
| engine | TensorFold native engine, DeepSeek-V4.1 two-node build | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-zig`, commit `3ff3b2a` (a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) |
| engine patches | three local commits, see [patches/](patches/) | `0001` is bertholomus's `5940208` (the no-GPU rank-cache writer) on this branch; `0002` (random seed per request, top_k handling) and `0003` (rank-cache tools take one or two ranks) are ours |
| recipe and kernel kit | `kit/` v0.6.1, run commands | [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) tag v0.6.1 (Apache-2.0) |
| weights | EXL3 2.9 bpw (mul1 codebook): routed experts 3 bpw (2 bpw on layers 18-22), attention 5 bpw, shared experts 4-5 bpw, MTP/DSpark dense parts 6 bpw, head 6 bpw, indexer `wk` 8 bpw | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (MIT) |
| Engram tables | shards 47 and 48 of the 1017 base checkpoint (2 x 101.5 GB) on each node | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT) |
| container | build and run | `nvcr.io/nvidia/pytorch:26.07-py3` (CUDA 13.3), Zig 0.17.0 |
| chat template | the one from the TP4 run, byte-exact against DeepSeek's `encoding.py` | [../2026-10-17-1017-tensorfold/launch/chat_template.jinja](../2026-10-17-1017-tensorfold/launch/chat_template.jinja) |

Credits are in [CREDITS.md](../../CREDITS.md).

## Findings so far

- **No kernel had to be built.** The pack's trellis widths are K = 2, 3, 4, 5, 6 and 8 ([results/k-coverage-2.9bpw.txt](results/k-coverage-2.9bpw.txt)); the recipe's kit has dense kernels for 1 to 8 bits and expert kernels for the half-bit ranges `(2,10)`, `(2,16)` and `(8,8)`. The MTP/DSpark dense tensors, 6 bpw here, are covered.
- **The loader plan accepts the pack** for both ranks (`tf-dsv41-rank-cache --index-only`), 106.19 GB (98.9 GiB) per rank.
- **Kit manifest: 222 of 223 files match**; the one difference is the image-routing bias file, which is new in the 1017 weights (same as the TP4 lane).
- **The four-node TP4 engine patches are not needed**: the `deepseek-v41-zig` branch already has the Triton variant fallback, the indexer alignment fix, the configurable RDMA GID index (`TF_RDMA_GID_INDEX`) and the whole-extent expert loader.
- **Rank caches are made without a GPU** (about 6 minutes a rank on one node reading the pack), [launch/inputs_tp2.sh](launch/inputs_tp2.sh); the second rank's file is copied over the fabric.

## Reproduce

Node order (rank): Reddie 0, Bluey 1. Fabric: rail 1 `enp1s0f0np0` 192.168.192.0/24, HCA `rocep1s0f0`; rail 2 `enP2p1s0f0np0` / `roceP2p1s0f0` 192.168.193.0/24 ([launch/rail2.sh](launch/rail2.sh), runtime addresses); RoCE v2 GID index 3.

1. **Image**: `docker pull nvcr.io/nvidia/pytorch:26.07-py3` on both nodes.
2. **Engine**: [launch/setup.sh](launch/setup.sh) (Zig, engine at `3ff3b2a`, recipe), then `git cherry-pick 5940208` and `git am patches/*.patch` (resolve the cherry-pick's README and kit-tool conflicts in favour of the branch); [launch/run_build.sh](launch/run_build.sh) + [launch/build.sh](launch/build.sh): about 2 minutes, no GPU attached.
3. **Weights**: `HF_HUB_DISABLE_XET=1 hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --local-dir <MODELS>/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` on one node, plus the Engram shards 47/48 of the base checkpoint on both.
4. **Kit**: the recipe's `kit/` at v0.6.1; add the four RoPE tables (`kit/tools/make_rope.py`, or the Hugging Face copy of the kit) and the 1017 `vision/gate_bias_vl.safetensors` (`kit/tools/make_bias_vl.py <base checkpoint>`); `kit/tools/verify_kit.sh` reports 222 of 223.
5. **Runtime model dir** (same path on both nodes): `config.json`, `tokenizer*.json`, `model.safetensors.index.json`, `exl3_k_map.json`, `quantization_config.json`, `encoding/`, the pack's shard 43 (the vision tower), `chat_template.jinja`.
6. **Rank caches** ([launch/inputs_tp2.sh](launch/inputs_tp2.sh)): `node_inputs.sh MODEL CACHE RANK 2` per rank, then copy rank 1's `rank-cache/rank1of2-*.bin` and the token map to the other node.
7. **Serve**: [launch/config.tp2.env](launch/config.tp2.env) and [launch/serve_tp2.sh](launch/serve_tp2.sh) (via [launch/start_tp2.sh](launch/start_tp2.sh)); logs in `LOG_DIR/tp2-lane.log`. [launch/precheck_tp2.sh](launch/precheck_tp2.sh) checks memory, containers, caches and files before a start.
8. **Bench**: [../2026-10-17-1017-tensorfold/tools/v41bench_tf.py](../2026-10-17-1017-tensorfold/tools/v41bench_tf.py) (same prompt set and method as the TP4 run); gates with `sanity.py`, `fleet_check.py`, `concurrent_check.py`, `sampling_check.py`, `long_check.py` from the same folder; the 69-scenario eval with [tools/run_eval69_tf_tp2.sh](tools/run_eval69_tf_tp2.sh).

## Results (deployment gates done; bench and eval running)

| | TP2, 2 Sparks (this lane) | TP4, 4 Sparks (3.5 bpw, `tf1017b-rail2`) |
|---|---|---|
| weights per node | 98.9 GiB | about 64 GiB |
| load time (s) | 27.1 | 107 to 120 |
| context window | 1,048,576 | 1,048,576 |
| streams | 4 | 16 |
| KV pool (positions, from the startup log) | 1,048,576 (shared by the 4 streams) | 2,097,152 |
| memory left per node while serving | 6.7 GiB (rank 0) / 8.2 GiB (rank 1) | 28 to 41 GiB |
| C1 aggregate tok/s | 73.1 | 93.6 |
| C2 / C4 | 106.0 / 135.8 | 136.9 / 189.3 |
| TTFT C1 (s) | 0.24 | 0.19 |
| cold prefill 11.6K / 46.8K / 93.3K tok/s | 2,180 / 2,200 / 2,135 (186.8K: 2,044) | 2,140 / 2,181 / 2,215 |
| longest needle found | 131,073 (2,059 tok/s prefill; longer pending) | 500,000 |
| 69-scenario eval quality | TBD | 97.1 |

Bench tables and raw JSON: [results/tp2a/](results/tp2a/) (prompt set v1, C1-C4 because the lane runs 4 streams; chart `speed_sheet.png`).

## Layout

- [DECISION.md](DECISION.md): engine, kit, nodes and why. [RUNBOOK_PHASE_B.md](RUNBOOK_PHASE_B.md): the commands for the measured run. [STATUS.md](STATUS.md).
- [patches/](patches/): the three engine commits (`git am`, after the cherry-pick above).
- [launch/](launch/): scripts and configs as run. [tools/](tools/): the TP2 eval wrapper (the other tools are in the TP4 run folder). [results/](results/), [logs/](logs/).
