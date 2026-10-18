# DeepSeek-V4.1-Flash-1017 with every matrix in EXL3 (LibertAI 3.5 bpw), vLLM TP4 on 4x DGX Spark

2026-10-17 / 18. One bring-up, one boot, benchmarked against the native 1017 lane ([`runs/2026-10-17-1017`](../2026-10-17-1017/README.md), boot `1017g`).

**Result: it loads and answers correctly on the first boot, uses much less memory, and decodes about 0.70-0.75x as fast as the native lane. The speed gap is mostly speculative decoding: the DSpark draft accepts 12.4 % of its tokens (1.62 tokens per step) against 53 % (3.65) on the native lane.** The 69-scenario eval was not run (see "What was not done").

| | native 1017g | **full EXL3 (this run)** |
|---|---|---|
| Model memory per Spark | 81.7 GiB | **56.2 GiB** |
| KV pool (tokens, 300K context) | 1,523,740 | **3,701,062** (2.4x) |
| Model loading time (head) | 288 s | **208 s** |
| Boot to serving | about 11 min (`restore_1017.sh`) | **374 s** |
| C1 / C6 aggregate tok/s | 48.5 / 165.7 | **36.0 / 115.8** (0.74x / 0.70x) |
| Draft tokens accepted | 53 % (3.65 tokens/step) | **12.4 % (1.62 tokens/step)** |
| Correctness checks (count-to-100, code, tool call, one image, served names) | pass | pass |

## The checkpoint

[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) (MIT, 239.9 GiB, EXL3 trellis with the `mul1` codebook, 3.53 bpw). Unlike the checkpoints behind the earlier EXL3 lanes (EXL3 routed experts, native FP8 everything else) it stores **every matrix** in EXL3:

- routed experts 4 bpw on layers 0-9 and 30-39, 3 bpw on 10-29, one K per layer;
- shared experts, `wq_a`, `wq_b`, `wkv`, `wo_b`, KV compressor `wkv`/`wgate`, indexer `wq_b`, Engram `wkv`, head and the MTP/DSpark drafter at 6 bpw; indexer `wk` at 8 bpw;
- `wo_a` as 8 per-group slices `wo_a.slice.0..7` of [4096 -> 1024];
- not in the repo: the two Engram n-gram tables, read unquantized from shards 47 and 48 of the base 1017 checkpoint.

Its model card says the MTP/DSpark drafter was "quantized without calibration data". The download was verified against the HF tree (53 files, 239.95 GiB, 192,495 tensors, safetensors headers consistent with file sizes) by [`tools/verify_download.py`](tools/verify_download.py); the Engram shards 47/48 match the sizes in the model card.

## What it took in the loader

[`patch/exl3-1017-full/`](../../patch/exl3-1017-full/CHANGES.md): our patch set on top of `dsv41-exl3-sr4`. In short:

- `wo_a` slices: a new grouped EXL3 module, each tensor-parallel rank holding the 2 groups of its own heads, plus an `o_proj` that keeps the inverse-RoPE step without FP8 quantization;
- shared experts in EXL3, padded from 2304 to 2560 columns (5 whole 128-blocks per rank), because an EXL3 slice has to start and end on a 128 boundary (a split at 576 gives 158-161 % error);
- the head with a vocabulary padded to 129,536 for the same reason;
- checkpoint-name aliases and the default fused-module map in `exl3_config.py` (the multimodal wrapper declares none);
- the compressor and indexer `wk` taking the EXL3 config, with the compressor score GEMM kept on the default stream (the cuda-exl3 activation workspace is process-wide, so two EXL3 GEMMs must not overlap on different streams; for the same reason `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` in the go script);
- `DSV41_ENGRAM_BASE_DIR` so the Engram reader takes the tables from the base checkpoint.

Checks before the boot, none of which needed the serving lane:

- a pure-PyTorch EXL3 reference decoder ([`tools/exl3_ref.py`](tools/exl3_ref.py), derived from the kernel sources and ExLlamaV3's format definitions) reproduces the base weights: 2 % relative error at 6 bpw, 7.9 % at 4 bpw, 14.4 % at 3 bpw, 0.65 % at 8 bpw ([report](results/oracle-check-report.md));
- CPU-only import and name-resolution checks against the real `quantization_config.json`, including the drafter modules recovered from the shard headers ([`tools/test_exl3_names.py`](tools/test_exl3_names.py));
- once the lane was stopped, a single-GPU test of the real layer classes with tensor-parallel ranks emulated, against the oracle ([`tools/gpu_unit_test_1017x.py`](tools/gpu_unit_test_1017x.py), [output](results/gpu-unit-test.txt)): wq_b, wo_b, fused wq_a/wkv, all four ranks of `wo_a`, the padded shared expert, the vocabulary-parallel head (argmax agreement 1.00), the compressor including the gate-less layer 20, indexer `wk`/`wq_b` and Engram `wkv` all land at 0.04-0.35 % relative error.

## Bring-up log

| time (UTC) | event |
|---|---|
| 02:18:42 | native lane stopped on all four Sparks (`stop_lane_1017x.sh`, logs saved) |
| 02:19 | single-GPU unit test passes |
| 02:19:40 | `restore_1017x.sh 1017x`: prelaunch gate passes (GPU burn 87-89 TFLOPS on all four) |
| 02:25:54 | **serving**, 374 s after the kick; weights 193 s on the head, draft reload 7 s; CUDA graphs captured (FULL_AND_PIECEWISE, 1.58 GiB + 0.42 GiB); served names `deepseek-v4.1-flash`, `glm-5.3-flash`; 300K; vision on |
| 02:26-02:30 | screen bench |
| 02:32-02:39 | full bench |
| 02:40:01 | lane stopped to free the Sparks for the next run |

No boot failed. The run uses `boot1017x-go.sh`: the `1017g` levers (NCCL 8 channels, Engram fast staging and 128 read threads, b12x RoCE all-reduce/all-gather, indexer prefill TP-split, `expandable_segments:False`, DSpark k=5, 300K, gmu 0.80) plus the shared-expert stream switch above.

## Benchmark against 1017g

Same `v41bench.py`, prompt set `v1`, thinking off ([`results/boot1017x`](results/boot1017x/report.md), [`results/compare-vs-1017g.md`](results/compare-vs-1017g.md)). Native numbers from [`runs/2026-10-17-1017/results/boot1017g`](../2026-10-17-1017/results/boot1017g/).

| C | aggregate tok/s, 1017g | EXL3 | ratio | per-stream, 1017g | EXL3 |
|---|---|---|---|---|---|
| C1 | 48.5 | 36.0 | 0.74 | 53.8 | 37.8 |
| C2 | 76.6 | 57.6 | 0.75 | 44.9 | 31.1 |
| C3 | 102.6 | 76.0 | 0.74 | 39.7 | 27.7 |
| C4 | 123.2 | 90.5 | 0.73 | 36.5 | 24.8 |
| C5 | 144.6 | 102.0 | 0.71 | 34.4 | 22.9 |
| C6 | 165.7 | 115.8 | 0.70 | 32.3 | 21.3 |

Per-stream tok/s at C1 by category:

| category | 1017g | EXL3 | ratio |
|---|---|---|---|
| coding | 76.3 | 42.3 | 0.55 |
| json | 63.2 | 25.0 | 0.40 |
| math | 56.2 | 50.7 | 0.90 |
| reasoning | 55.6 | 35.8 | 0.64 |
| format | 75.8 | 45.1 | 0.59 |
| summary | 35.8 | 46.7 | 1.30 |
| prose | 36.1 | 29.1 | 0.81 |
| narrative | 31.7 | 27.6 | 0.87 |
| counting (ceiling, not a headline) | 87.1 | 41.1 | 0.47 |

Cold prefill (unique prompt, 1-token reply):

| prompt tokens | 1017g tok/s | EXL3 tok/s |
|---|---|---|
| 2,950 | 1,903 | 1,759 |
| 11,592 | 1,936 | 1,279 |
| 46,810 | 2,194 | 1,822 |
| 93,335 | 2,201 (42.4 s) | 1,512 (61.7 s) |

The 11.6K and 46.8K EXL3 cells are from the screen run on the fresh boot: in the full bench those two prompts hit the prefix cache from the screen run (TTFT under 0.3 s), so their full-bench cells are not prefill speeds.

## Where the speed went

- **Speculative decoding.** The server's `SpecDecoding` log lines over the whole run give 18,432 accepted of 144,085 drafted tokens (12.8 %); over the full bench window alone 12.4 %, 1.62 tokens per step. An early snapshot (9,827 drafts) has position-0 acceptance of 20.6 %, then 16.3, 12.1, 10.1 and 8.6 % for positions 1-4; the native lane averages 3.65 tokens per step. On trivially predictable text (counting) the draft still reaches 5.5-6.0 of 6, so the path works end to end; the draft's distribution is what is off. The categories where the native lane gains most from speculation (json 0.40x, counting 0.47x, coding 0.55x, format 0.59x, reasoning 0.64x) are the ones that fall furthest.
- **Step time.** Where speculation gives little on either lane, EXL3 is close to native: summary 1.30x, math 0.90x, narrative 0.87x, prose 0.81x. The dense EXL3 GEMMs (attention, shared expert, head) therefore cost at most about 0-20 % per step here, not a factor of two. (The screen run's prose 23.7 tok/s was a cold-JIT number: the first requests after a boot trigger Triton compiles, see the `jit_monitor` lines in [`results/boot1017x/head.log`](results/boot1017x/head.log).)
- **Prefill** is 8-34 % slower than native (the 93K prompt 61.7 s against 42.4 s, 11.6K 1,279 against 1,936 tok/s).
- The earlier experts-only EXL3 lane (FP8 attention, native drafter) reached C6 190 and prose 41 on the September checkpoint; it is the reference for what the same hardware does with a native drafter.

What this run did not establish: whether the low acceptance comes from the drafter's weights (6 bpw without calibration), from something in how the drafter is loaded or fed, or from the target's hidden states differing from the native ones. Open leads, in the order we would test them: serve the same checkpoint with the native 1017 drafter tensors (position-0 acceptance should return to about 85 % if the EXL3 drafter weights are the cause); compare the draft's logits for one fixed prompt between the native and EXL3 drafters on one GPU; then profile one decode step for the dense EXL3 ops. A follow-up profile found the dense EXL3 GEMMs at the memory-bandwidth roofline, so the decode gap comes from the drafter's acceptance rather than the dense layers; the drafter investigation is still open.

## What was not done

- **The 69-scenario eval** was not run on this lane. The run was handed over before it started; the eval command is `tools/run_eval69_exl3.sh`.
- **No GPU microbench of the EXL3 dense ops** against FP8.
- **The prefix-cached full-bench prefill cells** are not valid prefill speeds (see above).

## Reproduce

On a machine that can ssh to the four Sparks, from the repo root:

```
bash runs/2026-10-17-1017-exl3/tools/stage_1017x.sh      # patch set dsv41-exl3-1017x, launcher, fan-out, test tools on all four
# on Reddie as root:
bash /root/stop_lane_1017x.sh                            # stops vllm_dsv41 everywhere, saves logs (tools/stop_lane_1017x.sh)
bash ~/kai-exl3-1017/gpu_check_1017x.sh                  # single-GPU loader unit test (optional)
bash /root/restore_1017x.sh 1017x                        # prelaunch gate, boot, postcheck (tools/restore_1017x.sh, launch/boot1017x-go.sh)
bash /root/bench1017.sh boot1017x full
```

Rolling back to the native lane: `bash /root/restore_1017.sh 1017g` (about 11 minutes).

## Files

- `launch/`: `dsv41-tp4-1017x.sh` (per-rank launcher), `boot_dsv41_1017x.sh` (fan-out), `boot1017x-go.sh`.
- `tools/`: staging, stop, prelaunch/restore, download verification, name and GPU tests, the oracle decoder, bench comparison, eval wrapper, results collection.
- `results/`: `boot1017x` (full bench, postcheck, head log), `boot1017x-screen`, `compare-vs-1017g.md`, `gpu-unit-test.txt`, `oracle-check-report.md`.
