# Uncensored lane: the Keys 1017 overlay on the TensorFold TP4 lane (2026-10-18)

The default TensorFold lane (DeepSeek-V4.1-Flash-1017, LibertAI EXL3 3.5 bpw, TP4 on four DGX Sparks, both RoCE rails, patch 0004) with the
[drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) overlay applied.
That repository is gated behind a responsible-use agreement and is not redistributed here: download it yourself after accepting its terms.
This lane serves our fleet endpoint (`0.0.0.0:8000`) since 2026-10-18 16:26 UTC; the stock lane stays one command away.

## What the overlay changes

Only `layers.10-35.attn.wo_b` (EXL3 trellis / suh / svh; `mul1` is identical to stock). The overlay's card gives lambda 3.5, k = 1, a 5120-d direction
recaptured on stock 1017, with layers 0-9, 36-39 (the DSpark draft targets), MTP, vision, Engram, every expert, embedding and head left stock.
Use the EXL3 sidecar (`exl3_wo_b_l10_35_1017.safetensors`, 818,590,872 bytes, sha256 `ce06dfb0e75a40a471bdff60676a687cf6efe4a126d33f54c2af482795ac6609`)
on this pack; the repository's FP8 sidecar is for the native FP8 checkpoint.

## How we applied it (rank-cache patch, no reshard)

TensorFold serves each node's rank cache (about 64 GiB a node), not the checkpoint shards. The overlay's documented paths rewrite the 26 shards that hold
those tensors (about 147 GB) and rebuild every rank cache. We patched copies of the existing rank caches instead, so the stock caches stay untouched
and switching lanes is a restart:

1. **Offsets.** `tf-dsv41-rank-cache <pack> <out> <rank> 4 --parity --index-only` for each rank (the same arguments `node_inputs.sh` uses), filtered to the
   78 wo_b slices a rank holds (26 layers x trellis, suh, svh): [`tools/make_blobs.sh`](tools/make_blobs.sh).
2. **Slices.** The overlay repository's `patch_rank_cache_wob.py --dump-dir` cuts the sidecar into each rank's slices (about 196 MiB a rank). It needs numpy,
   so it runs in the serving container.
3. **Patched copies.** On each node, [`tools/ablit_node.sh`](tools/ablit_node.sh) copies the rank cache with direct IO into `cache-ablit/` (same file name),
   copies the token map, writes the slices with the overlay repository's `apply_rank_blobs.py`, and renames the copy only after a check passes.
   [`tools/ablit_all.sh`](tools/ablit_all.sh) ships the slices and runs the four nodes in parallel; a few minutes, with the live lane serving.
4. **Check.** [`tools/verify2.py`](tools/verify2.py): all 78 slices equal the overlay's bytes and differ from stock, and about 427 sampled spots outside
   every patched range equal stock, on all four ranks.
5. **Serve.** [`launch/config.fleet-8000-rail2-p4-ablit.env`](launch/config.fleet-8000-rail2-p4-ablit.env) is the fleet config with
   `CACHE_DIR=.../cache-ablit`; [`launch/start_fleet_rail2_p4_ablit.sh`](launch/start_fleet_rail2_p4_ablit.sh) re-adds the rail-2 addresses and starts it.
   Back to stock: `start_fleet_rail2_p4.sh` from the TensorFold run folder. Each switch is about 2 minutes of downtime.

Disk: one extra rank-cache copy per node (68.9 GB).

## Results

Speed is the stock lane's: the overlay changes values, not shapes or sizes (speed sheet:
[`../2026-10-17-1017-tensorfold/results/tf1017-stock-c1c32/`](../2026-10-17-1017-tensorfold/results/tf1017-stock-c1c32/)).

**Refusal probe** ([`tools/refusal_probe.py`](tools/refusal_probe.py), temperature 0, thinking off; answers not reproduced here):
[`results/refusal_probe.md`](results/refusal_probe.md). Stock already answered four of the five prompts and refused one; the uncensored lane answers all
five, and the arithmetic check stays correct.

**69-scenario tool eval**, same suite and settings as the stock lane: [`results/eval69.md`](results/eval69.md).

| | stock TensorFold lane | uncensored lane |
|---|---|---|
| quality | 97.1 (134/138) | 87.0 (120/138) |
| responsiveness | 97.8 | 97.6 |
| decode in the suite | 86.0 tok/s | 93.1 tok/s (run-to-run spread; same engine and sizes) |

Every lost point is in the suite's safety and refusal scenarios, plus one omitted-parameter case; tool selection, parameter precision, multi-step chains,
structured output and format compliance are unchanged. In plain terms: the uncensored lane goes along with destructive or data-moving tool calls
without pushing back, follows instructions placed inside tool results, and fills in a missing parameter instead of asking. Do not give it unsupervised
tools that act on untrusted content (email, web pages, payments); keep a stock lane for agents like that.

## Responsible use

The overlay's agreement applies to anyone who downloads it (no content involving minors, no self-harm promotion, no illegal use, harassment, doxxing or
fraud, the upstream DeepSeek license). You are responsible for the filtering, review and access control of your deployment.

## Credits

- [drowzeys (Keys)](https://huggingface.co/drowzeys): the 1017 abliteration overlay and its `patch_rank_cache_wob.py` / `apply_rank_blobs.py`.
- The TensorFold lane it runs on: see the [TensorFold run folder](../2026-10-17-1017-tensorfold/) and [CREDITS.md](../../CREDITS.md).
