# DeepSeek-V4.1-Flash on two DGX Sparks: TensorFold TP2 with image prompt caching and chunked prefill

DeepSeek-V4.1-Flash-1017 served by [TensorFold](https://github.com/ashhart/TensorFold)'s native engine across **two**
DGX Sparks (GB10, tensor parallel over the direct QSFP link), with LibertAI's 2.9 bpw EXL3 pack: 1M-token context,
4 streams, an OpenAI-compatible API with image input.

This is a trimmed fork of [tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks).
It keeps only that repository's TP2 lane. Its two daily-use fixes started as local engine patches and are now upstream in
`bertholomus/TensorFold`'s `deepseek-v41-zig` branch, so the lane runs branch head and carries four patches (0001-0004) instead of seven:

1. **A long chat with an image re-read its whole context every turn.** The engine's prompt cache matched kept states by
   token ids, and every image is its placeholder token repeated, so any request holding an image skipped the cache. Now
   kept states are keyed by image content (`3de79d83`, was patch `0004`).
2. **One long cold prompt stalled everything else on the pair.** The DeepSeek prompt pass ran whole, so streams already
   decoding and short new requests waited for it. Upstream runs the pass one 2,048-row chunk at a time with the other
   streams' rounds between (`a27d2f00`, was patches `0006`-`0007`), switched on with `TF_FILL_CHUNKS=1`; patch `0004`
   re-ports what upstream did not take — a prompt that arrives mid-pass joins it, and the shortest waiting prompt goes
   first (was patch `0005`).

## What changed, why, and what it did

One table for the whole delta from upstream, so the reason for every choice and its measured effect sit together. The
throughput numbers come from one session on one freshly rebooted pair (next section), so they are comparable.

| Change | Why | Impact |
|---|---|---|
| Engine pin `3ff3b2a` &rarr; branch head `2aa3c47`: 26 upstream commits, drop patches 0004-0007 | tool-call parsing, prompt-cache correctness on long sessions, chat-template edge cases, buffer leaks | replies identical by `token_sha`; throughput neutral on decode and prefill (next section) |
| Patch `0001` | each node's weight file and token map built from the checkpoint; the kit's packer/generators | the two-node build and rank caches; needed to serve |
| Patch `0002` | opt-in per-request sampling seed and `top_k` handling (`TF_SEED_RANDOM`, `TF_TOP_K_MAX`) | reproducible sampling when asked for; no effect at defaults |
| Patch `0003` | `tf-dsv41-rank-cache` and `tf-dsv41-check` take one or two ranks (the four-node parity split is not here) | rank-cache building on this pair; needed to serve |
| Patch `0004` | upstream took the chunked pass (`a27d2f00`) but not shortest-waiting-first scheduling | a prompt arriving mid-pass joins it, shortest waiting goes first: 0.84 s TTFT sent 2 s into a 44K pass; 213 unit tests pass |
| `TF_FILL_CHUNKS=1` | one long cold pass ran whole and stalled every other stream on the pair | other streams keep decoding; prefill costs nothing (whole pass within 0.3% of chunked) |
| `TF_DS_ENGRAM_LEAD=1` | on the old pin each Engram layer read its next chunk's rows twice (1,411 tok/s at 8K) | pre-reboot +11-13% (8K 1,411&rarr;1,569, 32K 1,413&rarr;1,602); after a reboot every config already reads 2,400+, so it costs nothing and only pays on a fragmented boot |
| `--max-tokens 131072` | thinking counts against the cap, so 32,768 could truncate a long turn before its answer | no throughput cost; long reasoning turns fit |
| `--prompt-cache-gib 3` | the 1.5 GiB default fills on long sessions and starts evicting | more follow-ups hit cache; no throughput cost; 7.2 / 9.0 GiB free per rank against a 4 GiB floor |
| Reboot both nodes before a benchmark or a launch (P0-4) | Unmovable pages fragment after days of uptime: 238-242 GB/s against 262 GB/s | prefill 1,411-1,602 &rarr; 2,400-2,460 tok/s on every config, including the old engine; decode unchanged |
| EAQ v1 2.9 bpw pack &mdash; rejected | same size, +0.36 points of top-1 agreement and &minus;0.0017 nats KL | below the resolution of the 69-scenario eval; costs a 196 GiB re-download and a draft-head bits change that can lower acceptance, so it is not taken |

## Results

Measured on two DGX Sparks with this lane's settings, greedy decoding. Every reply below was identical between the
stock and the patched engine (the numbers are from the `3ff3b2a` pin; on branch head the same behaviour comes from the
upstream commits, and the exact-reply gates are re-run for each engine change.)

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
| engine | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) branch `deepseek-v41-zig` at `2aa3c47` (branch head, 2026-10-23; a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) + [patches/](patches/) `0001`-`0004` |
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
- `hf` and `rsync` on rank 0. DGX OS ships `rsync`; for `hf`: `python3 -m venv BASE/venv && BASE/venv/bin/pip install huggingface_hub`
  and put `BASE/venv/bin` on `PATH`.

## Reproduce

Everything lives under `BASE` on both nodes: `/var/tmp/tf1017-tp2`, or the folder in `TP2_BASE` when it is set (export it
before step 2 and keep it set for the scripts). Commands run on rank 0 unless noted.

1. **Image** (both nodes): `docker pull nvcr.io/nvidia/pytorch:26.07-py3`.
2. **Engine**: clone this repository on rank 0 and run [launch/setup.sh](launch/setup.sh). It fetches Zig 0.17.0 and
   bertholomus/TensorFold at `2aa3c47`, applies `patches/*.patch` with `git am`, and copies the launch scripts and a
   template `config.tp2.env` into `BASE`.
3. **Config**: edit `BASE/config.tp2.env`: the two nodes' user and address on the QSFP link, the RoCE HCAs and netdev
   (`ibdev2netdev`), and the RoCE v2 GID index (see the comments). Only rank 0 reads it.
4. **Build**: `BASE/run_build.sh` (the build container, no GPU, a few minutes), then `BASE/install_engine.sh` to put the
   binaries in `BASE/engine-bin` on both nodes.
5. **Weights** (rank 0): `HF_HUB_DISABLE_XET=1 hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --local-dir <PACK>`.
6. **Engram**: `hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir BASE/engram`,
   then `rsync -a BASE/engram/ <rank 1>:BASE/engram/` (each node reads its own copy).
7. **Kit**: `hf download bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10 --include 'kit/*' --local-dir BASE/kit-dl && mv BASE/kit-dl/kit BASE/kit`.
   The 1017 update needs its own vision gate bias: `python3 BASE/engine/tools/dsv41-zig/kit/make_bias_vl.py <1017 base checkpoint dir> BASE/kit/vision/gate_bias_vl.safetensors`
   (standard library only; it reads only the `*.ffn.gate.bias_vl` tensors, from the shards the index names). Then
   `bash BASE/kit/tools/verify_kit.sh` (a download keeps no execute bits): it reports `1 of 223 files missing or different`,
   `vision/gate_bias_vl.safetensors`, the 1017 bias that replaced the kit's. Then `rsync -a BASE/kit/ <rank 1>:BASE/kit/`.
8. **Model dir** (`BASE/model`): from the pack, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
   `model.safetensors.index.json`, `exl3_k_map.json`, `quantization_config.json`, `encoding/` and
   `model-00043-of-00043.safetensors` (the vision tower); plus [launch/chat_template.jinja](launch/chat_template.jinja) as
   `chat_template.jinja` (it replaces the pack's). Then `rsync -a BASE/model/ <rank 1>:BASE/model/`.
9. **Rank caches**: `PACK=<PACK> BASE/inputs_tp2.sh` builds both ranks' weight files and the token map (CPU only, low
   priority, about 20 minutes) and moves rank 1's to rank 1.
10. **Serve**: `BASE/start_tp2.sh` checks both nodes ([launch/precheck_tp2.sh](launch/precheck_tp2.sh)), then starts rank 1
    and rank 0 in containers named `tp2-lane` and waits for a chat reply. The API is on rank 0, port 8000, model
    `deepseek-v4.1-flash`; the log is `BASE/logs/tp2-lane.log`. `BASE/stop_tp2.sh` stops both.

To keep it up across reboots, run `start_tp2.sh` from a systemd unit on rank 0, or a loop that restarts the lane when
`/v1/models` stops answering (stop it with `serve_tp2.sh stop` first: the precheck refuses a start while either node
still has a `tp2-lane` container).

## Engine pin and patches

`launch/setup.sh` checks out the branch-head pin `2aa3c47` and applies `patches/0001`-`0004` as commits; `launch/run_build.sh`
builds it and `launch/install_engine.sh` installs into `BASE/engine-bin`. Nothing in that path overwrites an existing
build elsewhere.

To verify the lane rather than trust it, run [launch/gates_tp2.py](launch/gates_tp2.py) against the live pair: 12 text
prompts drafted, concurrent == solo, 9 image prompts alone, 18 image requests four at a time — all compared by
`tensorfold.token_sha`; then restart with `--no-drafts` and require drafted == serial, token for token.

## Verification (2026-10-23)

Engine `2aa3c47` + patches `0001`-`0004`, with `TF_FILL_CHUNKS=1`, `TF_DS_ENGRAM_LEAD=1`, `--max-tokens 131072` and
`--prompt-cache-gib 3`.

| Gate | Result |
|---|---|
| drafted == serial, 12 text prompts | 12/12 identical (`token_sha`) |
| concurrent == solo, 12 prompts | 12/12 identical |
| images alone | 9/9 answered |
| images four at a time, 18 requests | 18/18 identical to their solo replies |
| a short prompt 2 s into a 44K-token cold pass | TTFT 0.84 s (gate: under 10 s) |

Measured on the live lane afterwards:

- A short question sent 2 s into a 44K-token cold pass came back in 0.84 s with the pass still running — patch `0004`
  admits it and takes the fewest-rows-left prompt first (the old lane's 1.2 s behaviour). Decoders keep their rounds
  through a long pass (median gap 43 ms, longest 1.38 s over a 39.5K pass).
- Throughput is unchanged by the engine change; the numbers and their spread are in the four-config section below.

## The 1.4K prefill number, explained (2026-10-23)

The audit flagged a 78K cold prompt reported at 1,413 tok/s against the recipe's 2,135 at 93.3K. Measured on this lane, before the 2026-10-23 reboot,
with the recipe's own `kit_bench.py`, warm, fixed words:

| Config | 8K | 32K |
|---|---|---|
| as served before this test (`TF_FILL_CHUNKS=1`) | 1,411 | 1,413 |
| `TF_DS_ENGRAM_LEAD=1`, same chunks | **1,571** | **1,600** |
| `TF_DS_ENGRAM_LEAD=1`, whole pass (no chunks) | 1,567 | 1,605 |
| `TF_DS_ENGRAM_LEAD=1`, `TF_FILL_CHUNKS=4` | 763 | 1,259 |

Three answers, all measured before the reboot:

1. **The gap is not the prompt content or the chunking.** Natural prose and random words measure the same; a whole pass
   is within 0.3% of the one-chunk pass. Larger pieces are much worse, so one chunk stays.
2. **The gain is upstream `45c25b1b`, behind an opt-in flag on head.** That commit made each Engram layer read its next
   chunk's rows ahead; at `45c25b1b` it is unconditional and measures 1,570 / 1,603. A later commit (`9673a5ee`) made it
   opt-in, so on branch head the flag `TF_DS_ENGRAM_LEAD=1` is what turns it on; without the flag head reads 1,413.
   The flag is now in `config.tp2.env` and the 12-prompt gate is token-identical with it on (12/12 `token_sha`).
3. **The clocks are not latched:** 2.48–2.50 GHz at 60–64 W under a dense 40K cold prompt pre-reboot, and 2.41–2.50 GHz at 68–83 W on the fresh boot, so no unplug is needed.

Before the reboot the lane sat at ~1,600 tok/s at 32K against the recipe's 2,189 (27%). The bisect bracketed the
loadable range: `3ff3b2a` = 1,411, `45c25b1b` = 1,603, head = 1,602 — no commit between our old pin and head moves prefill.
The recipe's engine (`ce20a545`) builds but cannot load this lane's kit (`VerbsFailed`: it predates the 1017 pack's kernel
verbs), so it looked like the figure came from another engine + kit + pack era. In other words: the 1.4K was real and
fixed by `45c25b1b` + the flag; the remaining 27% turned out to be memory fragmentation, see the correction below.

**Corrected 2026-10-23, after the reboot.** The 27% was not another era: it was memory fragmentation. On a freshly
rebooted pair every config, including `3ff3b2a`, prefills at 2,400-2,460 tok/s at 8K and 32K, above the recipe's 2,189.
Before the reboot the same lane read 1,411-1,602. The bisect still stands — no commit between the old pin and head moves
prefill — and the lead flag only helps on a fragmented boot. See the four-config comparison below.


## Four-config comparison, one boot (2026-10-23)

The earlier throughput columns were not comparable: each was measured in its own session and the decode protocol was
never recorded. This run fixes that. Both nodes were rebooted, then all four P0 configs were measured in one session on
that boot, twice, with the round order reversed. Protocol: 256-token replies on the engine's own
`tensorfold.tokens_per_second`; C1 three reps, C2/C4 two reps; prefill `kit_bench.py` at 8K/32K, three reps with the cold
first rep discarded; a warm-up pass after every restart.

| Config | decode C1 | decode C2 | decode C4 | prefill 8K | prefill 32K |
|---|---|---|---|---|---|
| Original: `3ff3b2a` + 0001-0007, cache 1.5, max-tokens 32,768 | 71.4 ± 3.7 | 104.1 ± 3.2 | 142.1 ± 6.4 | 2,409 ± 2 | 2,460 ± 7 |
| P0-1: head + 0004 + `TF_FILL_CHUNKS=1` | 67.0 ± 5.6 | 108.3 ± 8.9 | 144.2 ± 2.0 | 2,416 ± 3 | 2,460 ± 2 |
| P0-1 + `TF_DS_ENGRAM_LEAD=1` | 69.9 ± 4.8 | 99.4 ± 5.7 | 137.0 ± 9.0 | 2,403 ± 7 | 2,461 ± 2 |
| + `--max-tokens 131072`, `--prompt-cache-gib 3` | 74.8 ± 5.6 | 98.7 ± 5.3 | 148.4 ± 8.2 | 2,409 ± 8 | 2,458 ± 3 |

Every difference between columns is inside one standard deviation. Three conclusions:

1. **The engine upgrade and both flags are throughput-neutral.** Nothing regressed; nothing got faster either, on decode
   or prefill.
2. **The prefill gain was the reboot, not the flag.** Pre-reboot the same lane read 1,411 (no lead) to 1,602 (lead);
   after the reboot every config, lead or not, reads 2,400-2,460 — above the recipe's 2,189. The lead flag is free
   insurance that only pays under fragmentation.
3. **The old 97 tok/s decode reading does not reproduce for any config.** The old engine reads 71.4 C1 on this boot,
   the same as head, so no config regressed against it; the 97 belongs to a session with an unrecorded protocol.

Correctness on the same lane: 12/12 replies identical to the Original recording (`token_sha`), drafted == serial 12/12,
concurrent == solo 12/12, images 9/9 alone and 18/18 at four a time.

**The EAQ v1 pack is rejected.** Its paired gain is 0.36 points of top-1 agreement and −0.0017 nats KL at the same size,
which will not show on the 69-scenario eval, and it costs a 196 GiB re-download plus a draft-head bits change that may
lower acceptance — and acceptance is what decode rides on. Expected value is negative; keep the current pack.

## Tested

These steps were run end to end on two DGX Sparks into a fresh `TP2_BASE` beside a running lane (2026-10-22): the
engine built from this repository's patches is byte-identical in source to the one measured above, the kit verified as
described in step 7, the rank caches came out under the same hashed file names as the running lane's, the precheck passed, the lane started and the
image follow-up benchmark gave the same replies and cache hits as in Results. Not exercised: `hf download` of the pack
and the Engram shards (existing copies were used) and `make_bias_vl.py` (its output from an earlier run was reused).

## Checking that it works

The engine logs one line per prompt pass:

```
prompt cache: 78343 tokens, resumed at 77824, kept 0; ...
done chatcmpl-... prompt=78343 cached=77824 ...
```

In a long chat holding an image, a follow-up turn should show `cached` close to `prompt` (keyed by the images' digests;
on the stock engine it stays 0). With `TF_FILL_CHUNKS=1`, a long cold prompt's pass is logged in pieces, and other
streams keep decoding between them.

## Layout

- [patches/](patches/): the four engine commits applied in order to the branch-head pin by `setup.sh`. `0001`-`0003` are
  the lane's own; `0004` re-ports the old `0005`'s shortest-waiting-first scheduling onto upstream's chunked pass. The
  old `0004` (image-keyed prompt states) and `0006`-`0007` (the chunked pass) are upstream as `3de79d83`, `a27d2f00`.
- [launch/](launch/): setup, build, install, inputs, the lane's config template, its start, stop and precheck scripts,
  and the exact-reply gates.
- [NOTICE](NOTICE), [CREDITS.md](CREDITS.md), [third_party/TensorFold/](third_party/TensorFold/): attribution and the
  engine's license and notice texts.

## License

This repository's own files are MIT ([LICENSE](LICENSE), from the upstream repository). The patches are changes to
TensorFold and are Apache-2.0, as is the engine they build; see [NOTICE](NOTICE). The weights are MIT. Credits for
everyone this lane builds on are in [CREDITS.md](CREDITS.md).
