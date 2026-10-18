"""Single-GPU unit test of the full-EXL3 loader path against a pure-PyTorch oracle.

Builds the real vLLM layer classes the model uses (ColumnParallel / RowParallel / fused Merged / Replicated linears, the shared
expert MLP with its padding, the vocabulary-parallel EXL3 head, the grouped wo_a), loads real checkpoint tensors through the same
weight_loader calls the model's load_weights makes, emulates tensor parallel rank r of 4 by patching the rank/size getters, and
compares each layer's output (concatenated or summed over the 4 emulated ranks) with x @ W_eff computed by the reference decoder
in exl3_ref.py (pure torch, derived from the CUDA kernel sources). Needs one free GPU (the serving lane stopped).

Run in the vllm-dsv41:exl3b-roce image with --gpus all, the dsv41-exl3-1017x patch set mounted per mounts.txt, the checkpoint at
/models/ckpt and this folder + exl3_ref.py at /kx.
"""
import contextlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/kx")
import exl3_ref as R  # noqa: E402
from safetensors import safe_open  # noqa: E402

MD = "/models/ckpt"
WM = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
_H = {}


def T(name):
    f = WM[name]
    if f not in _H:
        _H[f] = safe_open(f"{MD}/{f}", framework="pt", device="cpu")
    return _H[f].get_tensor(name)


def W_eff(mod):
    """Reference effective weight [in, out] fp32 on cuda for checkpoint module `mod`."""
    w = R.dequant_exl3(T(mod + ".trellis"), T(mod + ".suh"), T(mod + ".svh"), mul1=T(mod + ".mul1"))
    return w.float().cuda()


# ---- vLLM bring-up (single process, one GPU) ----
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402

VC = VllmConfig()
CTX = set_current_vllm_config(VC)
CTX.__enter__()
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method="tcp://127.0.0.1:29571", backend="nccl")
initialize_model_parallel(1, 1)

import vllm.model_executor.layers.linear as L_  # noqa: E402
import vllm.model_executor.layers.vocab_parallel_embedding as V_  # noqa: E402
import vllm.models.deepseek_v4.nvidia.model as M4_  # noqa: E402
import vllm.models.deepseek_v4_1.nvidia.model as M41_  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    ColumnParallelLinear, MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # noqa: E402
from cuda_exl3.config import Exl3Config  # noqa: E402

TPN = 4
_cur = {"rank": 0, "size": 1}


def _patch(rank, size):
    _cur["rank"], _cur["size"] = rank, size
    for m in (L_, V_, M4_, M41_):
        m.get_tensor_model_parallel_rank = lambda: _cur["rank"]
        m.get_tensor_model_parallel_world_size = lambda: _cur["size"]


qd = json.load(open(f"{MD}/quantization_config.json"))
QC = Exl3Config(qd)
QC.packed_modules_mapping = {}      # as under the multimodal wrapper class
print("Exl3Config:", len(QC.modules), "modules; shared_expert_pad_block", QC.shared_expert_pad_block)

dev = "cuda"
torch.manual_seed(1234)
fails = []


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-9))


def report(name, got, ref, tol):
    e = rel(got, ref)
    ok = e < tol and torch.isfinite(got.float()).all().item()
    print(f"{'ok  ' if ok else 'FAIL'} {name:46s} rel_err={e:.4f} (tol {tol})")
    if not ok:
        fails.append(name)


def load_plain(layer, ckpt_mod, shard_id=None, names=("trellis", "suh", "svh", "mul1")):
    for leaf in names:
        p = getattr(layer, leaf)
        w = T(f"{ckpt_mod}.{leaf}")
        if shard_id is None:
            p.weight_loader(p, w)
        else:
            p.weight_loader(p, w, shard_id)


def finish(layer):
    layer.quant_method.process_weights_after_loading(layer)


def rand_x(t, k, dtype=torch.bfloat16):
    return (torch.randn(t, k, device=dev) * 0.5).to(dtype)


TOK = [1, 6, 48]

# ---- 1. column-parallel wq_b at TP4 ----
LAY = 5
W = W_eff(f"layers.{LAY}.attn.wq_b")
for t in TOK:
    x = rand_x(t, 1280)
    outs = []
    for r in range(TPN):
        _patch(r, TPN)
        lin = ColumnParallelLinear(1280, 32768, bias=False, quant_config=QC, return_bias=False,
                                   prefix=f"language_model.model.layers.{LAY}.attn.wq_b")
        load_plain(lin, f"layers.{LAY}.attn.wq_b")
        finish(lin)
        outs.append(lin(x))
        del lin
    report(f"wq_b column-parallel TP4, {t} tok", torch.cat(outs, dim=1), x.float() @ W, 0.03)
del W

# ---- 2. row-parallel wo_b at TP4 (sum of ranks) ----
W = W_eff(f"layers.{LAY}.attn.wo_b")
for t in TOK:
    x = rand_x(t, 8192)
    acc = 0
    for r in range(TPN):
        _patch(r, TPN)
        lin = RowParallelLinear(8192, 5120, bias=False, quant_config=QC, reduce_results=False,
                                return_bias=False, prefix=f"language_model.model.layers.{LAY}.attn.wo_b")
        load_plain(lin, f"layers.{LAY}.attn.wo_b")
        finish(lin)
        acc = acc + lin(x[:, r * 2048:(r + 1) * 2048]).float()
        del lin
    report(f"wo_b row-parallel TP4, {t} tok", acc, x.float() @ W, 0.03)
del W

# ---- 3. fused wq_a|wkv (replicated, two shards with their own suh) ----
_patch(0, 1)
Wa, Wk = W_eff(f"layers.{LAY}.attn.wq_a"), W_eff(f"layers.{LAY}.attn.wkv")
lin = MergedColumnParallelLinear(5120, [1280, 512], bias=False, quant_config=QC, disable_tp=True,
                                 prefix=f"language_model.model.layers.{LAY}.attn.fused_wqa_wkv")
load_plain(lin, f"layers.{LAY}.attn.wq_a", 0)
load_plain(lin, f"layers.{LAY}.attn.wkv", 1)
finish(lin)
for t in TOK:
    x = rand_x(t, 5120)
    out, _ = lin(x)
    report(f"fused_wqa_wkv, {t} tok", out, torch.cat([x.float() @ Wa, x.float() @ Wk], dim=1), 0.03)
del lin, Wa, Wk

# ---- 4. wo_a slices through Exl3GroupedLinear at TP4 ----
from vllm.models.deepseek_v4_1.exl3_dense import Exl3GroupedLinear  # noqa: E402
Wg = [W_eff(f"layers.{LAY}.attn.wo_a.slice.{g}") for g in range(8)]
for t in TOK:
    xg = (torch.randn(t, 2, 4096, device=dev) * 0.5).to(torch.bfloat16)
    for r in range(TPN):
        _patch(r, TPN)
        g = Exl3GroupedLinear(QC, f"language_model.model.layers.{LAY}.attn.wo_a", 8, 2, r, 4096, 1024)
        for i in range(2):
            gi = 2 * r + i
            for leaf in ("trellis", "suh", "svh", "mul1"):
                p = getattr(g.slice[str(gi)], leaf)
                p.weight_loader(p, T(f"layers.{LAY}.attn.wo_a.slice.{gi}.{leaf}"))
        out = g(xg)
        ref = torch.cat([xg[:, i, :].float() @ Wg[2 * r + i] for i in range(2)], dim=1)
        if t == TOK[-1] or r == 0:
            report(f"wo_a grouped rank {r}, {t} tok", out, ref, 0.03)
        del g
del Wg

# ---- 5. shared expert at TP4: padded gate_up / down, clamped SwiGLU ----
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP  # noqa: E402
S = f"layers.{LAY}.ffn.shared_experts"
W1, W3, W2 = W_eff(f"{S}.w1"), W_eff(f"{S}.w3"), W_eff(f"{S}.w2")
for t in TOK:
    x = rand_x(t, 5120)
    acc = 0
    for r in range(TPN):
        _patch(r, TPN)
        mlp = DeepseekV4MLP(5120, 2304, "silu", swiglu_limit=10.0, quant_config=QC, reduce_results=False,
                            prefix=f"language_model.model.{S}")
        pad = M41_.DeepseekV4Model._pad_exl3_shared_expert_weight
        for leaf in ("trellis", "suh", "svh", "mul1"):
            for w, sid in (("w1", 0), ("w3", 1)):
                p = getattr(mlp.gate_up_proj, leaf)
                p.weight_loader(p, pad(f"{S}.{w}.{leaf}", T(f"{S}.{w}.{leaf}"), 128), sid)
            p = getattr(mlp.down_proj, leaf)
            p.weight_loader(p, pad(f"{S}.down_proj.{leaf}", T(f"{S}.w2.{leaf}"), 128))
        finish(mlp.gate_up_proj)
        finish(mlp.down_proj)
        acc = acc + mlp(x).float()
        del mlp
    g_, u_ = x.float() @ W1, x.float() @ W3
    ref = (torch.nn.functional.silu(g_.clamp(max=10.0)) * u_.clamp(-10.0, 10.0)) @ W2
    report(f"shared expert (padded 2560) TP4, {t} tok", acc, ref, 0.05)
del W1, W3, W2

# ---- 6. vocabulary-parallel EXL3 head at TP4 (padding 512) ----
Wh = W_eff("head")
for t in (1, 6):
    x = rand_x(t, 5120)
    parts = []
    for r in range(TPN):
        _patch(r, TPN)
        head = ParallelLMHead(129280, 5120, padding_size=128 * TPN, quant_config=QC,
                              prefix="language_model.lm_head")
        for leaf in ("trellis", "suh", "svh", "mul1"):
            p = getattr(head, leaf)
            p.weight_loader(p, T(f"head.{leaf}"))
        finish(head)
        parts.append(head.quant_method.apply(head, x))
        if r == 0 and t == 1:
            print("head: padded vocab", head.num_embeddings_padded, "per rank", head.num_embeddings_per_partition)
        del head
    logits = torch.cat(parts, dim=1)[:, :129280]
    report(f"lm_head vocab-parallel TP4, {t} tok", logits, x.float() @ Wh, 0.03)
    am_ok = (logits.float().argmax(-1) == (x.float() @ Wh).argmax(-1)).float().mean().item()
    print(f"     argmax agreement {am_ok:.2f}")
del Wh

# ---- 7. compressor kv|gate (fp16 in / fp32 out), single-shard ratio-1 compressor, indexer, engram ----
from vllm.models.deepseek_v4_1.exl3_dense import exl3_linear_apply  # noqa: E402
_patch(0, 1)
for cl, shards in ((2, ("wkv", "wgate")), (20, ("wkv",))):
    sizes = [512] * len(shards)
    lin = MergedColumnParallelLinear(5120, sizes, bias=False, return_bias=False, quant_config=QC, disable_tp=True,
                                     prefix=f"language_model.model.layers.{cl}.attn.compressor.fused_wkv_wgate")
    refs = []
    for i, s in enumerate(shards):
        load_plain(lin, f"layers.{cl}.attn.compressor.{s}", i)
        refs.append(W_eff(f"layers.{cl}.attn.compressor.{s}"))
    finish(lin)
    x = rand_x(6, 5120)
    out = exl3_linear_apply(lin, x, torch.float32)
    report(f"compressor layer {cl} ({'+'.join(shards)}) fp32 out", out, torch.cat([x.float() @ w for w in refs], 1), 0.03)
    assert out.dtype == torch.float32
    del lin

lin = ReplicatedLinear(512, 128, bias=False, quant_config=QC, prefix="language_model.model.layers.2.attn.indexer.wk")
load_plain(lin, "layers.2.attn.indexer.wk")
finish(lin)
x = rand_x(6, 512)
report("indexer.wk 8 bit", lin(x)[0], x.float() @ W_eff("layers.2.attn.indexer.wk"), 0.02)
del lin
lin = ReplicatedLinear(1280, 4096, bias=False, quant_config=QC, prefix="language_model.model.layers.2.attn.indexer.wq_b")
load_plain(lin, "layers.2.attn.indexer.wq_b")
finish(lin)
x = rand_x(6, 1280)
report("indexer.wq_b", lin(x)[0], x.float() @ W_eff("layers.2.attn.indexer.wq_b"), 0.03)
del lin
lin = ReplicatedLinear(6144, 25600, bias=False, quant_config=QC, return_bias=False, prefix="language_model.model.layers.1.engram.wkv")
load_plain(lin, "layers.1.engram.wkv")
finish(lin)
x = rand_x(6, 6144)
report("engram.wkv", lin(x), x.float() @ W_eff("layers.1.engram.wkv"), 0.03)
del lin

print("RESULT", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
