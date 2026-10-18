"""CPU-only check of the patched cuda_exl3/config.py name handling for the full-EXL3 DeepSeek-V4.1 checkpoint.

Run inside the vllm-dsv41:exl3b-roce image with the dsv41-exl3-1017x patch set mounted (no GPU needed):
    python3 test_exl3_names.py [path/to/quantization_config.json]
Without an argument a synthetic tensor_storage with the checkpoint's module names is used; with the real file the same
assertions run against the shipped names.
"""
import json
import sys

from cuda_exl3.config import Exl3Config, CB_MUL1


def entry(name, k, n, bits):
    return {
        "quant_format": "exl3",
        "mul1_multiplier": 2212286765,
        "stored_tensors": {
            f"{name}.trellis": {"shape": [k // 16, n // 16, 16 * bits]},
            f"{name}.suh": {"shape": [k]},
            f"{name}.svh": {"shape": [n]},
            f"{name}.mul1": {"shape": []},
        },
    }


def synthetic():
    st = {}
    for L in range(40):
        a = f"layers.{L}.attn"
        st[f"{a}.wq_a"] = entry(f"{a}.wq_a", 5120, 1280, 6)
        st[f"{a}.wkv"] = entry(f"{a}.wkv", 5120, 512, 6)
        st[f"{a}.wq_b"] = entry(f"{a}.wq_b", 1280, 32768, 6)
        st[f"{a}.wo_b"] = entry(f"{a}.wo_b", 8192, 5120, 6)
        for g in range(8):
            st[f"{a}.wo_a.slice.{g}"] = entry(f"{a}.wo_a.slice.{g}", 4096, 1024, 6)
        f = f"layers.{L}.ffn"
        for w, (k, n) in (("w1", (5120, 2304)), ("w3", (5120, 2304)), ("w2", (2304, 5120))):
            st[f"{f}.shared_experts.{w}"] = entry(f"{f}.shared_experts.{w}", k, n, 6)
            for e in (0, 1):
                st[f"{f}.experts.{e}.{w}"] = entry(f"{f}.experts.{e}.{w}", k, n, 4 if (L < 10 or L >= 30) else 3)
    for L in (2, 8, 14, 20):
        c = f"layers.{L}.attn.compressor"
        st[f"{c}.wkv"] = entry(f"{c}.wkv", 5120, 512, 6)
        if L != 20:
            st[f"{c}.wgate"] = entry(f"{c}.wgate", 5120, 512, 6)
        st[f"layers.{L}.attn.indexer.wk"] = entry(f"layers.{L}.attn.indexer.wk", 512, 128, 8)
    for L in (2, 8, 14, 20, 24, 28, 32, 36):
        st[f"layers.{L}.attn.indexer.wq_b"] = entry(f"layers.{L}.attn.indexer.wq_b", 1280, 4096, 6)
    for L in (1, 14):
        st[f"layers.{L}.engram.wkv"] = entry(f"layers.{L}.engram.wkv", 6144, 25600, 6)
    st["head"] = entry("head", 5120, 129280, 6)
    st["mtp.0.main_proj"] = entry("mtp.0.main_proj", 15360, 5120, 6)
    for M in range(3):
        a = f"mtp.{M}.attn"
        st[f"{a}.wq_a"] = entry(f"{a}.wq_a", 5120, 1280, 6)
        st[f"{a}.wkv"] = entry(f"{a}.wkv", 5120, 512, 6)
        st[f"{a}.wq_b"] = entry(f"{a}.wq_b", 1280, 32768, 6)
        st[f"{a}.wo_b"] = entry(f"{a}.wo_b", 8192, 5120, 6)
        for g in range(8):
            st[f"{a}.wo_a.slice.{g}"] = entry(f"{a}.wo_a.slice.{g}", 4096, 1024, 6)
        f = f"mtp.{M}.ffn"
        for w, (k, n) in (("w1", (5120, 2304)), ("w3", (5120, 2304)), ("w2", (2304, 5120))):
            st[f"{f}.shared_experts.{w}"] = entry(f"{f}.shared_experts.{w}", k, n, 6)
            for e in (0, 1):
                st[f"{f}.experts.{e}.{w}"] = entry(f"{f}.experts.{e}.{w}", k, n, 6)
    return {"version": "1.6.2", "codebook": "mul1", "tensor_storage": st,
            "original_quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                             "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                             "expert_dtype": "fp4"}}


import os
if os.environ.get("CUDA_EXL3_MODEL_PATH"):
    Exl3Config._model_path_hint = os.environ["CUDA_EXL3_MODEL_PATH"]   # lets the config recover the mtp.* modules from the shard headers
cfgd = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else synthetic()
if "original_quantization_config" not in cfgd:
    cfgd["original_quantization_config"] = synthetic()["original_quantization_config"]
c = Exl3Config(cfgd)
c.packed_modules_mapping = {}      # as for the multimodal wrapper class: nothing declared
fails = []


def want(prefix, n_infos, bits=None):
    infos = c.resolve(prefix)
    ok = infos is not None and len(infos) == n_infos and (bits is None or infos[0].bits == bits)
    print(("ok   " if ok else "FAIL ") + prefix, [i.name for i in infos] if infos else None)
    if not ok:
        fails.append(prefix)


W = "language_model.model."
want(W + "layers.5.attn.fused_wqa_wkv", 2, 6)
want(W + "layers.5.attn.wq_b", 1, 6)
want(W + "layers.5.attn.wo_b", 1, 6)
want(W + "layers.5.ffn.shared_experts.gate_up_proj", 2, 6)
want(W + "layers.5.ffn.shared_experts.down_proj", 1, 6)
want(W + "layers.2.attn.compressor.fused_wkv_wgate", 2, 6)
want(W + "layers.20.attn.compressor.fused_wkv_wgate", 1, 6)
want(W + "layers.2.attn.indexer.wk", 1, 8)
want(W + "layers.24.attn.indexer.wq_b", 1, 6)
want(W + "layers.1.engram.wkv", 1, 6)
want("language_model.lm_head", 1, 6)
for g in range(8):
    want(W + f"layers.3.attn.wo_a.slice.{g}", 1, 6)
# DSpark drafter: layers.40+i / main_proj. The shipped quantization_config.json has no mtp.* entries; the plugin recovers
# them from the safetensors headers when the model directory is available, so these only run when they are present.
has_mtp = any(n.startswith("mtp.") for n in c.modules)
print("drafter modules present:", has_mtp)
if has_mtp:
    want("model.layers.41.attn.fused_wqa_wkv", 2, 6)
    want("model.layers.42.ffn.shared_experts.down_proj", 1, 6)
    want("model.layers.40.attn.wo_a.slice.7", 1, 6)
    want("model.main_proj", 1, 6)
# things that must NOT resolve
for p in (W + "layers.5.attn.compressor.fused_wkv_wgate", W + "layers.3.ffn.gate", W + "embed_tokens"):
    infos = c.resolve(p)
    print(("ok   " if infos is None else "FAIL ") + "(absent) " + p)
    if infos is not None:
        fails.append(p)

moe = c.resolve_moe(W + "layers.12.ffn.experts")
print("ok   " if moe and moe[0].bits == 3 else "FAIL ", "moe layer 12 w1 bits", moe[0].bits if moe else None)
if not (moe and moe[0].bits == 3):
    fails.append("moe12")
if has_mtp:
    moe = c.resolve_moe("model.layers.41.ffn.experts")
    print("ok   " if moe and moe[0].bits == 6 else "FAIL ", "moe drafter bits", moe[0].bits if moe else None)
    if not (moe and moe[0].bits == 6):
        fails.append("moe-mtp")

print("shared_expert_pad_block", c.shared_expert_pad_block, "n_backbone_layers", c.n_backbone_layers)
if c.shared_expert_pad_block != 128 or c.n_backbone_layers != 40:
    fails.append("attrs")
try:
    print("weight_block_size (delegated)", c.weight_block_size)
except AttributeError as e:
    print("FAIL weight_block_size", e)
    fails.append("wbs")

from vllm.models.deepseek_v4_1.exl3_dense import exl3_resolves, exl3_slices_resolve
assert exl3_resolves(c, W + "layers.5.attn.wq_b")
assert exl3_slices_resolve(c, W + "layers.5.attn.wo_a", 8)
assert not exl3_resolves(c, W + "layers.5.attn.compressor.fused_wkv_wgate")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
