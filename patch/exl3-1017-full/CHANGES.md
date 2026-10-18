# exl3-1017-full: loader changes for the full-EXL3 DeepSeek-V4.1-Flash-1017 checkpoint

Patch set `dsv41-exl3-1017x` = [`dsv41-exl3-sr4`](../../runs/2026-09-19-speedrun2/patches/sr4/CHANGES.md) (the EXL3 TP4 lane: cuda-exl3
overlays, b12x RoCE comm files, indexer prefill TP-split, draft shard filter) plus the files in this folder
(`build_patchset.sh` assembles it; `mounts.txt` is the full manifest).

The checkpoint is [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)
(MIT; EXL3 `mul1` codebook, 3.53 bpw, 239.9 GiB). Unlike the checkpoints the earlier EXL3 lanes used (EXL3 routed experts, native FP8
everything else) it stores **every matrix** as EXL3: attention, shared experts, indexer, compressor, Engram projection, head and the
MTP/DSpark drafter. Only the Engram n-gram tables are not in it; they come from the base 1017 shards 47 and 48.

## What each file changes

| file (mounted over) | change |
|---|---|
| `exl3_dense.py` (new, `models/deepseek_v4_1/`) | `Exl3GroupedLinear`: `wo_a` stored as `o_groups` EXL3 slices `wo_a.slice.<g>` of [4096 -> 1024]; a TP rank registers only the `o_groups / tp` slices of its own heads, under the checkpoint's names. `exl3_o_proj` (inverse RoPE without FP8 quantization, grouped `wo_a`, `wo_b`). `exl3_resolves` / `exl3_slices_resolve` / `exl3_linear_apply` helpers. |
| `exl3_config.py` (`cuda_exl3/config.py`) | Shared experts may be EXL3 (the old refusal is gone) and expose `shared_expert_pad_block = 128`. Checkpoint-name aliases: `lm_head` = `head`, shared `down_proj` = `w2`, drafter `layers.<40+i>` = `mtp.<i>`, `main_proj` = `mtp.0.main_proj`. Default packed-module map for DeepSeek-V4.1 (`gate_up_proj`, `fused_wqa_wkv`, `fused_wkv_wgate`), because the multimodal wrapper class declares none. A ratio-1 compressor (layer 20) has no gate, so its fused module resolves to `wkv` alone. Optional no-split-k method for shared-expert GEMMs (`DSV41_EXL3_SHARED_NOSPLIT=1`). |
| `attention.py` (`models/deepseek_v4_1/attention.py`) | `wo_a` built as `Exl3GroupedLinear` when the checkpoint has the slices; `indexer.wk` gets the EXL3 quant config when the checkpoint has it; the compressor score GEMM runs on the default stream after `fused_wqa_wkv` when it is EXL3 (the plugin's activation workspace is process-wide, so two EXL3 GEMMs must not overlap on different streams). |
| `compressor.py` | `fused_wkv_wgate` takes the EXL3 config when the checkpoint has it; `kv_score()` (fp16 in, fp32 out for EXL3, the original `torch.mm` otherwise). |
| `flashinfer_sparse.py` | `_o_proj` routes to `exl3_o_proj` for an EXL3 `wo_a`. |
| `dsv41_nvidia_model.py` (`models/deepseek_v4_1/nvidia/model.py`) | The head is built with the EXL3 config and a vocabulary padded to a multiple of 128 x tp (129280 -> 129536 at TP4: every rank's slice starts and ends on a 128-column Hadamard block; the 256 pad columns are two whole blocks zeroed through `svh`). Loader mapper rules for `head.{trellis,suh,svh,mul1}`. `wo_a` slices owned by other ranks are skipped. Shared-expert tensors are padded in checkpoint units (`_pad_exl3_shared_expert_weight`). |
| `dsv4_nvidia_model.py` (`models/deepseek_v4/nvidia/model.py`) | The shared-expert MLP rounds its intermediate size (2304) up to whole 128-blocks per rank (2560 at TP4) when the config asks for it. |
| `dspark.py` | Same shared-expert padding flag and `wo_a` slice skipping for the drafter's loader. |
| `vl_model.py` | Mapper rules for the EXL3 head under the multimodal wrapper. |
| `engram.py` | `DSV41_ENGRAM_BASE_DIR`: where the Engram tables are read from when the model directory does not hold them. The node-local row copies still take precedence where they cover a rank. |

## Why the shapes work at TP4

EXL3 applies its input and output transforms to independent 128-element blocks, so a tensor-parallel slice must start and end on a
128 boundary (a split at 576 of 2304 gives 158-161 % error against the full result; see `runs/2026-10-17-1017-exl3/results/oracle-check-report.md`).

| tensor | full | per rank | note |
|---|---|---|---|
| `wq_b` (column) | 32768 | 8192 | 64 blocks |
| `wo_b` (row, input) | 8192 | 2048 | 16 blocks |
| `wo_a` | 8 groups | 2 groups | whole slices |
| shared expert gate/up (column), down (row) | 2304 | 640 | padded to 2560 = 4 x 5 blocks; rank 3 holds 384 real + 256 zero columns |
| routed experts | 2304 | 512 / 640 / 640 / 512 | existing uneven split in `exl3_moe.py` |
| head (vocabulary) | 129280 | 32384 | padded to 129536 |
| attention / indexer / compressor / Engram linears | | replicated | not split |

Padding is exact because it is always whole blocks with zero scales: a zero `svh` on padded output columns makes them exactly zero
whatever the trellis holds there; the matching padded input rows of the next matrix see exact zeros (zero `suh`).

## Not changed

Routed-expert MoE (`exl3_moe.py`), the draft shard filter, the RoCE comm files, the indexer TP-split, `sparse_*`, `weight_utils.py`,
`vocab_parallel_embedding.py`, `virtual_heads.py` are the `dsv41-exl3-sr4` files.
