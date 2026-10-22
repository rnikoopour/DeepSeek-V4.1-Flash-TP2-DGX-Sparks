# DeepSeek-V4.1-Flash on two DGX Sparks: TensorFold TP2 with image prompt caching and chunked prefill

DeepSeek-V4.1-Flash-1017 served by [TensorFold](https://github.com/ashhart/TensorFold)'s native engine across **two**
DGX Sparks (GB10, tensor parallel over the direct QSFP link), with LibertAI's 2.9 bpw EXL3 pack: 1M-token context,
4 streams, an OpenAI-compatible API with image input.

This is a trimmed fork of [tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks).
It keeps only that repository's TP2 lane and adds four engine patches that fix two things you hit in daily use:

1. **A long chat with an image re-read its whole context every turn.** The engine's prompt cache matched kept states by
   token ids, and every image is its placeholder token repeated, so any request holding an image skipped the cache.
   `0004` keys kept states by image content instead.
2. **One long cold prompt stalled everything else on the pair.** The DeepSeek prompt pass ran whole, so streams already
   decoding and short new requests waited for it. `0005`-`0007` run the pass a 2,048-row chunk at a time with the other
   streams' rounds between, the shortest waiting prompt first (a port of upstream TensorFold's chunked prefill).

## Results

Measured on two DGX Sparks with this lane's settings, greedy decoding. Every reply below was identical between the
stock and the patched engine.

**A ~78K-token chat with an image in its first message, follow-up turns** (time to first token):

| turn | stock | patched | prompt tokens from cache (patched) |
|---|---|---|---|
| T2 (+78K tokens, cold) | 54.7 s | 55.2 s | 0 |
| T3 | 54.9 s | **0.90 s** | 77,824 of 78,343 |
| T4 | 55.1 s | **0.87 s** | 77,824 of 78,358 |
| T4 with a different image in the first message | 55.0 s | 55.1 s | 0 (a different image never matches) |

Text-only chats are unchanged (0.76 s vs 0.73 s for a cached 78K follow-up).

**A cold 78K-token prompt L, with a stream D already decoding and a short question S sent 5 s into L's pass:**

| | stock | patched |
|---|---|---|
| S, time to first token | 49.7 s | **1.2 s** |
| D, longest pause between tokens | 54.6 s | **1.6 s** |
| L, time to first token | 54.6 s | 56.7 s |

An image prompt whose pass was interrupted three times by other requests gave the same reply as the same prompt run
alone on an empty cache.

The stock lane's other numbers (73.1 tok/s at 1 user, 135.8 at 4, 1M-token needle found, eval 97.1) are in the
[upstream TP2 run folder](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks/tree/ab2d381/runs/2026-10-18-1017-tensorfold-tp2).

## What runs

| piece | source |
|---|---|
| engine | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) branch `deepseek-v41-zig` at `3ff3b2a` (a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) + [patches/](patches/) `0001`-`0007` |
| kernel kit | `kit/` of [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10) (the Hugging Face mirror of [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) v0.6.1, generated files included) |
| weights | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (MIT, 196.3 GiB; 98.9 GiB of weights per node) |
| Engram tables | shards 47 and 48 of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT, 2 x 101.5 GB) on each node |
| container | `nvcr.io/nvidia/pytorch:26.07-py3` (CUDA 13.3), for the build and the run; Zig 0.17.0 |

## Requirements

- Two DGX Sparks joined by a QSFP cable, RoCE working between them (`ibdev2netdev` shows the link's HCAs up).
- Rank 0 reaches both nodes (itself included) with `ssh -o BatchMode=yes` as a user in the `docker` group.
- On each node: about 110 GiB free memory with nothing else on the GPU, and Docker with the NVIDIA runtime. Disk: about
  600 GB on rank 0 (the 196 GB pack, the 203 GB Engram shards, its ~99 GB rank cache and rank 1's while it is built) and
  about 310 GB on rank 1. The pack can go once step 9 is done (the model dir keeps the files it needs).
- `hf` (huggingface_hub) and `rsync` on rank 0.

## Reproduce

Everything lives under `/var/tmp/tf1017-tp2` (`BASE`) on both nodes. Commands run on rank 0 unless noted.

1. **Image** (both nodes): `docker pull nvcr.io/nvidia/pytorch:26.07-py3`.
2. **Engine**: clone this repository on rank 0 and run [launch/setup.sh](launch/setup.sh). It fetches Zig 0.17.0 and
   bertholomus/TensorFold at `3ff3b2a`, applies `patches/*.patch` with `git am`, and copies the launch scripts and a
   template `config.tp2.env` into `BASE`.
3. **Config**: edit `BASE/config.tp2.env`: the two nodes' user and address on the QSFP link, the RoCE HCAs and netdev
   (`ibdev2netdev`), and the RoCE v2 GID index (see the comments). Only rank 0 reads it.
4. **Build**: `BASE/run_build.sh` (the build container, no GPU, a few minutes), then `BASE/install_engine.sh` to put the
   binaries in `BASE/engine-bin` on both nodes.
5. **Weights** (rank 0): `HF_HUB_DISABLE_XET=1 hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --local-dir <PACK>`.
6. **Engram** (both nodes): `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir BASE/engram`.
7. **Kit** (both nodes, same path): `hf download bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10 --include 'kit/*' --local-dir BASE/kit-dl && mv BASE/kit-dl/kit BASE/kit`.
   The 1017 update needs its own vision gate bias: `python3 BASE/engine/tools/dsv41-zig/kit/make_bias_vl.py <1017 base checkpoint dir> BASE/kit/vision/gate_bias_vl.safetensors`
   (standard library only; it reads only the `*.ffn.gate.bias_vl` tensors, from the shards the index names). Then
   `BASE/kit/tools/verify_kit.sh` (the replaced bias file is the one entry expected to differ from the kit's MANIFEST).
8. **Model dir** (both nodes, `BASE/model`): from the pack, `config.json`, `tokenizer*.json`, `model.safetensors.index.json`,
   `exl3_k_map.json`, `quantization_config.json`, `encoding/` and shard 43 (the vision tower); plus
   [launch/chat_template.jinja](launch/chat_template.jinja) as `chat_template.jinja`.
9. **Rank caches**: `PACK=<PACK> BASE/inputs_tp2.sh` builds both ranks' weight files and the token map (CPU only, low
   priority) and moves rank 1's to rank 1.
10. **Serve**: `BASE/start_tp2.sh` checks both nodes ([launch/precheck_tp2.sh](launch/precheck_tp2.sh)), then starts rank 1
    and rank 0 in containers named `tp2-lane` and waits for a chat reply. The API is on rank 0, port 8000, model
    `deepseek-v4.1-flash`; the log is `BASE/logs/tp2-lane.log`. `BASE/stop_tp2.sh` stops both.

To keep it up across reboots, run `start_tp2.sh` from a systemd unit on rank 0, or a loop that restarts the lane when
`/v1/models` stops answering (stop it with `serve_tp2.sh stop` first: the precheck refuses a start while either node
still has a `tp2-lane` container).

## Checking that the patches work

The engine logs one line per prompt pass:

```
prompt cache: 78343 tokens, resumed at 77824, kept 0; ...
done chatcmpl-... prompt=78343 cached=77824 ...
```

In a long chat holding an image, a follow-up turn should show `cached` close to `prompt`. On the stock engine it stays 0.

## Layout

- [patches/](patches/): the seven engine commits, applied in order to `3ff3b2a` by `setup.sh`.
- [launch/](launch/): setup, build, install, inputs, the lane's config template and its start, stop and precheck scripts.
- [NOTICE](NOTICE), [CREDITS.md](CREDITS.md), [third_party/TensorFold/](third_party/TensorFold/): attribution and the
  engine's license and notice texts.

## License

This repository's own files are MIT ([LICENSE](LICENSE), from the upstream repository). The patches are changes to
TensorFold and are Apache-2.0, as is the engine they build; see [NOTICE](NOTICE). The weights are MIT. Credits for
everyone this lane builds on are in [CREDITS.md](CREDITS.md).
