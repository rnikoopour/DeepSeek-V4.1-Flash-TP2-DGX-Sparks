# DeepSeek-V4.1-Flash-1017 on TensorFold (native Zig engine), TP4 on 4x DGX Spark, 2026-10-17/18

The 1017 update served by [TensorFold](https://github.com/ashhart/TensorFold)'s native engine with every matrix in EXL3 (LibertAI's 3.5 bpw pack), tensor-parallel over the same four Sparks as the vLLM lanes. This folder holds what is needed to reproduce the boot: patches, scripts with every env var, the benchmark commands and the results. Current state of each check is in [STATUS.md](STATUS.md).

The first boot worked: the four ranks loaded in 107.5 s and the lane answered correctly, with no new kernels needed. With our chat template loaded (second boot, 116.2 s) chat, tool calling, thinking and image input all pass; the 69-scenario eval scores 97.1 (native vLLM 1017g: 98.6); the recipe gates and the dual-rail test are still pending (see [STATUS.md](STATUS.md)).

## What runs

| piece | what | source |
|---|---|---|
| engine | TensorFold native engine, DeepSeek-V4.1 four-node (2D split) build | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-zig-tp4`, commit `5940208` (a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) |
| engine patches | three local commits on top, see [patches/](patches/) | `47a8910` and `b8ae0c7` are commits already public on the fork's `deepseek-v41-zig` branch; `TF_RDMA_GID` is ours |
| recipe, scripts, kernel kit | `scripts/serve.sh` (used unmodified), `kit/` (Triton + EXL3 kernel binaries), `tools/` | [bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10) tag v0.1.1 (Apache-2.0) |
| weights | all matrices EXL3 (mul1 codebook): routed experts 4 bpw on layers 0-9 and 30-39, 3 bpw on 10-29, everything else 6 bpw (indexer `wk` 8 bpw), 240 GB | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) (MIT) |
| Engram tables | shards 47 and 48 of the 1017 base checkpoint, a full local copy on every node (2 x 101.5 GB) | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT) |
| container | build and run | `nvcr.io/nvidia/pytorch:26.07-py3` (CUDA 13.3), Zig 0.17.0 |
| chat template | our own Jinja template, written from DeepSeek's `encoding/encoding.py` and byte-exact against it (10/10 official cases, 267 named cases, 8,000 random conversations, 5,098 cases through TensorFold's own Jinja engine); two small documented gaps in STATUS.md | [launch/chat_template.jinja](launch/chat_template.jinja) |

Credits for the people behind each piece are in [CREDITS.md](../../CREDITS.md) (TensorFold and its contributors; bertholomus for the fork, recipe and kernel kit; Lucas Peters for `b8ae0c7`; ecohash-co for the `TF_RDMA_GID` approach and the `_index_score` analysis in ashhart/TensorFold#559; LibertAI for the quantization; DeepSeek-AI for the model and the reference encoder; turboderp for the EXL3 format).

## Findings

- **The shipped kernel kit already covers the 3.5 bpw pack.** Expert kernels are templated on a half-bit range `(lo, hi)`: `(8, 8)` is exactly 4-bit, `(2, 10)` covers 3-bit and `(2, 16)` covers the 6-bit DSpark experts. Dense `linear`/`glinear` kernels exist for 2 to 8 bits (this pack's dense matrices are 6-bit, the indexer `wk` 8-bit). No kernel had to be built.
- **The loader plan accepts the pack as is.** `tf-dsv41-rank-cache --index-only` produced a valid plan for all four ranks (rank cache about 68.9 GB each) before anything was written, see [results/index-dry-run.txt](results/index-dry-run.txt).
- **The kit's `vision/gate_bias_vl.safetensors` differs from the recipe's MANIFEST, correctly.** `verify_kit.sh` reports 355 of 356 files matching: the 43 image-routing bias vectors are new in the 1017 weights (sha256 `4745da4b...`; identical whether copied from the base checkpoint or from the pack).
- **No chat template ships with the 1017 checkpoint**, so `/v1/chat/completions` first answered "this checkpoint has no chat template". We wrote one from DeepSeek's `encoding.py` ([launch/chat_template.jinja](launch/chat_template.jinja), copied into `MODEL_DIR`) and tested it against the reference encoder and TensorFold's own Jinja engine ([STATUS.md](STATUS.md)). The benchmark below was run before the template was loaded, so it uses `/v1/completions` with prompts encoded by DeepSeek's own `encoding.py` (chat mode, thinking off); the token ids are the ones the template produces.
- **Long prompts**: the failing lengths from ashhart/TensorFold#559 (131,073 to 149,796 tokens, the `_index_score` gap) pass with the Triton-variant fallback; the server logs the fallback once.

## Reproduce

Everything runs from Reddie (rank 0) as the user that owns the shared node-to-node ssh key; the other three are workers. Node order (rank): Reddie 0, Spark4 1, Asusi 2, Bluey 3 (ranks 0-1 and 2-3 are the 2D split's pairs). Fabric: rail 1 `enp1s0f0np0` 192.168.192.0/24, HCA `rocep1s0f0`, RoCE v2 GID index 3 (the recipe's default of 5 does not exist on these ports).

1. **Image**: `docker pull nvcr.io/nvidia/pytorch:26.07-py3` on all four nodes.
2. **Engine**: [launch/setup.sh](launch/setup.sh) fetches Zig 0.17.0 and clones the fork and the recipe; apply the patches (`git am patches/*.patch` on `5940208`); [launch/run_build.sh](launch/run_build.sh) + [launch/build.sh](launch/build.sh) build in the container (about 2 min): `zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121`. `ENGINE_DIR` = `tensorfold-native`, `tf-dsv41-lanes`, `tf-dsv41-rank-cache`, `node_inputs.sh`, `make_token_map.py` and `rdma_gather.fatbin` (the engine's own `dsv41_rdma` fatbin from the zig cache).
3. **Weights**: `HF_HUB_DISABLE_XET=1 hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw --local-dir <MODELS>/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw` (240 GB, on the head node, shared to workers over NFS), and the base 1017 checkpoint for its Engram shards.
4. **Engram**: [launch/copy_engram.sh](launch/copy_engram.sh) puts shards 47/48 on every worker's local NVMe (Reddie hardlinks them); the engine reads rows by offset, so the tables must be local.
5. **Model dir for the runtime** (`MODEL_DIR`, same path on every node): `config.json`, `tokenizer*.json`, `model.safetensors.index.json`, `encoding/`, the pack's shards 43 and 44 (the vision tower), and `chat_template.jinja`. [launch/distribute_model.sh](launch/distribute_model.sh).
6. **Kit**: `kit/` from the recipe, plus `launch/kitverify.sh` (makes the four RoPE tables and the bias file, then checks the manifest). Copy to `KIT_DIR` on every node ([launch/distribute.sh](launch/distribute.sh)).
7. **Per-node inputs** (no GPU, about 6 min, all four at once): [launch/inputs_r0.sh](launch/inputs_r0.sh) on the head and [launch/inputs_worker.sh](launch/inputs_worker.sh) `RANK` on the workers: `node_inputs.sh MODEL CACHE RANK` writes `rank-cache/rank<R>of4-<hash>.bin` (68.9 GB) and `dsv41_token_map.json`.
8. **Serve**: [launch/config.env](launch/config.env) is the recipe's `config.env` for this cluster (every env var is in it); [launch/start1.sh](launch/start1.sh) runs the recipe's `scripts/serve.sh start` (ranks 3, 2, 1, then 0, then a chat check). The lane runs as the `tp4-lane` container on each node; logs in `LOG_DIR/tp4-lane.log`. [launch/precheck.sh](launch/precheck.sh) checks free RAM, containers and rank caches before a boot.
9. **Bench**: [launch/run_bench.sh](launch/run_bench.sh) `LABEL [--levels ...] [--prefill ...]` runs [tools/v41bench_tf.py](tools/v41bench_tf.py), which is `/root/v41bench.py` (same prompt set `v1`, concurrency barrier, unique tags, usage-block token counts, cold prefill method) with one change: prompts are encoded with `encoding.py` and sent to `/v1/completions`. `--chat` switches to `/v1/chat/completions` (the template is loaded in the lane from the second boot on). [tools/sanity.py](tools/sanity.py) checks chat, the alias, a tool call and its round trip, thinking and image input on the chat route; [tools/run_eval69_tf.sh](tools/run_eval69_tf.sh) runs the 69-scenario eval. Long prompts: [tools/long_check.py](tools/long_check.py) (exact-length prompts sent as token ids, with a needle).

Memory per node while serving (16 streams, 1,048,576-token window, 2,097,152-position pool): model 71.87 GiB; 40.6 GiB available on the workers, 28-29 GiB on the head.

## Benchmark: TensorFold, rail 1 only (`tf1017a`)

Same prompt set and method as the vLLM runs ([README](../2026-10-17-1017/README.md)): temperature 0, thinking off, tokens from the server's usage block, cold prefill with a unique prefix. Cells are means over the 8 categories (counting excluded). Full tables: [results/tf1017a/](results/tf1017a/), [results/tf1017a-c8c16/](results/tf1017a-c8c16/).

| | native vLLM `boot1017g` | vLLM full-EXL3 (same pack) | **TensorFold, same pack** |
|---|---|---|---|
| C1 aggregate tok/s | 48.5 | 36.0 | **89.8** |
| C2 | 76.6 | | **132.8** |
| C3 | 102.6 | | **159.0** |
| C4 | 123.2 | | **180.3** |
| C5 | 144.6 | | **192.9** |
| C6 | 165.7 | 115.8 | **202.9** |
| C8 / C16 | | | **212.8 / 247.4** |
| TTFT C1 / C6 (s) | 0.28 / 0.45 | | 0.19 / 0.67 |
| cold prefill, 2,950 / 11,592 / 46,810 / 93,335 tokens (tok/s) | 1,903 / 1,937 / 2,194 / 2,201 | 1,759 at 2,950 (later rows hit the prefix cache) | 1,613 / 1,873 / 1,894 / 1,929 |
| 93,335-token prompt | 42 s | | 48.4 s |

Per-stream tok/s at C1 by category:

| category | native vLLM | vLLM full-EXL3 | **TensorFold** |
|---|---|---|---|
| code | 76.3 | 42.3 | **134.8** |
| format (Markdown table) | 75.8 | 45.1 | **139.4** |
| JSON | 63.2 | 25.0 | **107.6** |
| math | 56.2 | 50.7 | **119.9** |
| reasoning | 55.6 | 35.8 | **97.7** |
| prose | 36.1 | 29.1 | **78.0** |
| summary | 35.8 | 46.7 | **69.7** |
| narrative | 31.7 | 27.6 | **67.3** |
| counting (ceiling only) | 87.1 | 41.1 | 165.2 |

Reading it: decode is about 1.8x native vLLM at one stream and 22% more aggregate at six; cold prefill is about 12% slower than native vLLM 1017g (the engine's prefill was published at 2.8K tok/s using both rails; this run uses one, see STATUS.md).

Needle prompts (exact lengths, token ids, found the code every time): 20,000; 131,073; 145,152; 146,421; 149,796; 160,000 (83 s); 300,000 (162 s); 500,000 (285 s, 1,756 tok/s).

Higher concurrency (same set): C8 212.8 and C16 247.4 tok/s aggregate, 33.0 and 19.7 tok/s per stream, TTFT 0.85 and 1.63 s. The head node keeps 28-29 GiB available at 16 streams. 69-scenario eval (thinking off): quality 97.1 (134/138; 66 pass, 2 partial, 1 fail), responsiveness 97.8, 86.0 tok/s decode, against 98.6 and 43.0 tok/s on native vLLM 1017g; the one fail is a keyword false positive on a helpful answer ([STATUS.md](STATUS.md), [results/eval69](results/eval69/eval69-tf1017.md)). Engine unit tests: `zig build test` 212 of 213 (the container direct-I/O test fails, as the recipe's authors report). [results](logs/long_check1.log), [more](logs/long_check2.log).

## Layout

- [STATUS.md](STATUS.md): what is validated and what is pending.
- [patches/](patches/): the three engine commits (`git am`).
- [launch/](launch/): scripts and `config.env` exactly as run on Reddie and the workers.
- [tools/](tools/): the bench client, the long-prompt checker, the chat-route sanity checks, the 69-eval runner, the pack inventory script, the template test and its Zig parity harness.
- [results/](results/): benchmark outputs. [logs/](logs/): setup, build, kit verification, inputs, boot and long-prompt logs.
