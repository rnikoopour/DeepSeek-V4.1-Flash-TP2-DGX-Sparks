# SPDX-License-Identifier: Apache-2.0
"""Dense EXL3 helpers for the full-EXL3 DeepSeek-V4.1-Flash checkpoint.

The checkpoint stores every dense matrix of the model as EXL3 (trellis + suh +
svh + mul1 marker): attention, shared experts, indexer, compressor, Engram
projection, head and the MTP/DSpark drafter. Almost all of them go through the
stock `Exl3LinearMethod` of the cuda-exl3 plugin via ordinary vLLM linear
layers. Two things need code of their own and live here:

* `Exl3GroupedLinear`: the attention output projection `wo_a`. The checkpoint
  stores it as `o_groups` (8) independent [4096 -> 1024] matrices,
  `wo_a.slice.<g>.{trellis,suh,svh,mul1}`, one per output group. A tensor
  parallel rank owns `o_groups / tp` whole groups (heads of group g live on
  rank g * tp // o_groups), so each rank registers only its own slices, under
  the checkpoint's own names. Slices that belong to other ranks have no
  parameter and the model's load_weights skips them.
* `exl3_linear_apply` / `exl3_resolves`: small helpers used by the attention,
  indexer and compressor patches to decide whether a module is EXL3 in this
  checkpoint (the same model files also serve the FP8 checkpoints) and to run
  an EXL3 layer outside of `forward` (compressor scores, fp16 in / fp32 out).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

TILE = 16


def exl3_resolves(quant_config, prefix: str) -> bool:
    """True when `quant_config` is an EXL3 config that has a module for `prefix`.

    `prefix` is the vLLM module path of a (possibly fused) linear, for example
    `model.layers.2.attn.compressor.fused_wkv_wgate`; fused names resolve
    through the model's packed_modules_mapping inside the config.
    """
    if quant_config is None or getattr(quant_config, "get_name", lambda: "")() != "exl3":
        return False
    try:
        return quant_config.resolve(prefix) is not None
    except Exception:  # noqa: BLE001
        return False


def exl3_slices_resolve(quant_config, prefix: str, n_groups: int) -> bool:
    """True when every `<prefix>.slice.<g>` (g < n_groups) is an EXL3 module."""
    if quant_config is None or getattr(quant_config, "get_name", lambda: "")() != "exl3":
        return False
    try:
        return all(quant_config.resolve(f"{prefix}.slice.{g}") is not None
                   for g in range(n_groups))
    except Exception:  # noqa: BLE001
        return False


def exl3_linear_apply(layer: nn.Module, x: torch.Tensor,
                      out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Run an EXL3 vLLM linear layer on `x` without going through `forward`.

    With out_dtype=torch.float32 the activations go in as fp16 (the dtype the
    kernels compute in) so the result keeps fp16 precision before the final
    float() instead of being rounded to bf16 first.
    """
    method = layer.quant_method
    if out_dtype == torch.float32:
        return method.apply(layer, x.to(torch.float16)).float()
    out = method.apply(layer, x)
    return out if out_dtype is None else out.to(out_dtype)


class Exl3Slice(nn.Module):
    """One EXL3 matrix [in_features -> out_features] with its own parameters."""

    def __init__(self, in_features: int, out_features: int, bits: int, cb: int):
        super().__init__()
        dev = torch.cuda.current_device()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.cb = cb
        self.trellis = nn.Parameter(
            torch.empty(in_features // TILE, out_features // TILE, TILE * bits,
                        dtype=torch.int16, device=dev), requires_grad=False)
        self.suh = nn.Parameter(
            torch.empty(1, in_features, dtype=torch.half, device=dev),
            requires_grad=False)
        self.svh = nn.Parameter(
            torch.empty(out_features, dtype=torch.half, device=dev),
            requires_grad=False)
        # The codebook multiplier is a compile-time constant of the codebook id
        # for the kernels; it is still a tensor in the checkpoint.
        self.mul1 = nn.Parameter(
            torch.empty(1, dtype=torch.int32, device=dev), requires_grad=False)

        # Checkpoint tensors are [in] / [out] / scalar; suh is stored (1, in)
        # because the kernel takes one row per fused shard.
        def _load_suh(param, w):
            param.data[0].copy_(w)

        def _load_scalar(param, w):
            param.data.copy_(w.reshape(1))

        def _load_plain(param, w):
            param.data.copy_(w)

        self.trellis.weight_loader = _load_plain
        self.suh.weight_loader = _load_suh
        self.svh.weight_loader = _load_plain
        self.mul1.weight_loader = _load_scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from cuda_exl3 import ops

        return ops.exl3_linear(x, self.trellis.data, self.suh.data, self.svh.data,
                               [self.out_features], self.cb)


class Exl3GroupedLinear(nn.Module):
    """`wo_a` of a full-EXL3 checkpoint: `n_local_groups` independent matrices.

    Input  [tokens, n_local_groups, in_per_group] (inverse-RoPE'd attention
    output, one row block per group), output [tokens, n_local_groups * out].
    """

    is_exl3_grouped = True

    def __init__(self, quant_config, prefix: str, n_groups: int, n_local_groups: int,
                 tp_rank: int, in_per_group: int, out_per_group: int):
        super().__init__()
        self.n_groups = n_groups
        self.n_local_groups = n_local_groups
        self.in_per_group = in_per_group
        self.out_per_group = out_per_group
        self.first_group = tp_rank * n_local_groups
        # Registered under the checkpoint's names: wo_a.slice.<global group>.*
        self.slice = nn.ModuleDict()
        for i in range(n_local_groups):
            g = self.first_group + i
            infos = quant_config.resolve(f"{prefix}.slice.{g}")
            if infos is None or len(infos) != 1:
                raise ValueError(f"EXL3 {prefix}.slice.{g}: not found in the checkpoint")
            info = infos[0]
            if info.in_features != in_per_group or info.out_features != out_per_group:
                raise ValueError(
                    f"EXL3 {prefix}.slice.{g}: stored {info.in_features}x"
                    f"{info.out_features}, expected {in_per_group}x{out_per_group}")
            self.slice[str(g)] = Exl3Slice(in_per_group, out_per_group, info.bits, info.cb)
        # Size the plugin's shared workspaces for this shape now: they cannot
        # grow once CUDA graphs are being captured.
        try:
            from cuda_exl3 import ops
            from vllm.config import get_current_vllm_config

            sched = get_current_vllm_config().scheduler_config
            max_tokens = max(sched.max_num_batched_tokens, sched.max_num_seqs)
            first = self.slice[str(self.first_group)]
            ops.reserve(first.svh.data, max_tokens, in_per_group, out_per_group, 1)
        except Exception as e:  # noqa: BLE001  (no vLLM context in unit tests)
            logger.debug("EXL3 %s: workspace pre-reserve skipped (%s)", prefix, e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [tokens, n_local_groups, in_per_group] -> [tokens, n_local_groups*out]."""
        outs = []
        for i in range(self.n_local_groups):
            outs.append(self.slice[str(self.first_group + i)](x[:, i, :]))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


def exl3_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: "Exl3GroupedLinear",
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Attention output projection with an EXL3 `wo_a`.

    Same structure as the bf16-weight branch of `deep_gemm_fp8_o_proj`: inverse
    RoPE into [tokens, groups, group_width] (no FP8 quantization), the grouped
    wo_a, then wo_b (row-parallel, reduces across ranks).
    """
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    o_in, _ = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=128,
        tma_aligned_scales=False,
        quantize=False,
    )
    return wo_b(wo_a(o_in))
