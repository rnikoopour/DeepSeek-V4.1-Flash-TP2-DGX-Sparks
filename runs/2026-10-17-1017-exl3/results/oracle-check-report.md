# EXL3 reference decoder: validation report (DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)

All code in this directory, CPU-only torch, runs with `/opt/homebrew/bin/python3 -I <script>` (torch 2.10 / numpy / safetensors on the Mac mini; `exl3_ref.py` itself only needs torch + numpy).

## Result in one line

The reference decoder reproduces the base weights on the first try with no fitting: 6 bpw = 1.9-2.1 %, 4 bpw = 7.9 %, 3 bpw = 14.4 %, 8 bpw = 0.65 % relative Frobenius error (pure quantization noise). Every wrong convention I tried (transposed tile, suh/svh in a different place, no Hadamard, other MXFP4 nibble order) gives 137-4000 %.

## Files

| file | what |
|---|---|
| `exl3_ref.py` | the module: `dequant_exl3(trellis, suh, svh, mul1_marker, ...) -> W[in,out] fp32`, `exl3_linear_ref(x, trellis, suh, svh, mul1_marker, group_n=None, emulate_fp16=False)`, `iter_dequant_blocks` (128-row streaming), `dequant_inner` (trellis weight in the rotated basis), `unpack_states`, `codebook_lut`, `had128`. `python3 -I exl3_ref.py` runs the self test. |
| `base_ref.py` | safetensors raw reader (E8M0/FP8 safe), FP8-block / MXFP4 / BF16 base dequantizers (streamed per 128 input rows) |
| `check_layers.py` | tests 1-7 (tables below), writes `results_layers.json` |
| `check_wo_a.py` | all 8 wo_a slices against all 8 base row groups, writes `results_wo_a.json` |
| `check_structure.py` | conventions (a)(b), padding (c)(d), TP splits (e), writes `results_structure.json` |
| `pull.py`, `rssh.sh`, `rscp.sh` | pure-python byte-range extractor that ran on Reddie (no numpy/torch/safetensors there), ssh/scp wrappers |
| `upstream/` | ExLlamaV3 sources I read (see "Source of truth") |
| `data/` | the pulled tensors: `exl3_pulled.safetensors` (158 MB), `base_pulled.safetensors` (244 MB), `wo_a_slices.safetensors` (22 MB) |

Reddie: only header + byte-range reads (`nice -n 19`, ~0.5 GB total read), tiny output files written to `/var/tmp/exl3ref_*`, already deleted. No GPU, no repo write, no server touched. EXL3 shards used (2, 4, 14) were at their final size (`ls -l` repeated, identical to the other full-size shards).

## Source of truth

The scratchpad copy of the plugin has `gemm.cu`, `hadamard.cu`, `ops.py`, `linear.py` but NOT the three headers they include (`exl3_dq.cuh`, `exl3_common.cuh`, `exl3_had.cuh`; gemm.cu says they "come from ExLlamaV3 and are part of the on-disk format"). The tile/bit-stream/codebook/Hadamard definitions therefore come from upstream ExLlamaV3 master (`turboderp-org/exllamav3`: `exllamav3_ext/quant/{exl3_dq.cuh, codebook.cuh, reconstruct.cu, hadamard_inner.cuh}`, `modules/quant/exl3.py:get_weight_tensor`, saved in `upstream/`). gemm.cu calls the same entry point (`dq_dispatch<BITS, CB>(tile, lane << 3, frag_b[nb], frag_b[nb+1])`, `n_tiles_full` row stride, `(n_off+n0)/16` tile column) and the same epilogue order (`C -> half -> had128 -> svh`), and the numbers below close the loop empirically. The 'half_k' (x.5 bit) variants in upstream are not used by the plugin call and not present in this checkpoint (all integer bits).

## Decoding rules (derived, then confirmed by the numbers)

* `trellis` int16 `[K/16, N/16, 16*bits]` (K = in, N = out), row-major over tiles. One 16x16 tile = 8*bits little-endian uint32 words (adjacent int16 pairs, low half first) = a cyclic bit stream of 256*bits bits, read MSB-first inside each uint32, words in order.
* State for tile position p (0..255) = the 16-bit window that ENDS at stream bit `(p+1)*bits` (mod 256*bits, tail-biting), MSB first. Checked against a literal python port of `dq()` for bits 1..8 (`python3 -I exl3_ref.py`).
* Position to matrix element (mma.m16n8k16 B fragment, thread `lane` decodes positions 8*lane..8*lane+7):
  `p = 8*(4*(n%8) + (k%8)//2) + 4*(n//8) + 2*(k//8) + (k%2)`, k = row (input), n = column (output) inside the tile. Cross-checked against upstream `reconstruct.cu`'s shuffle to row-major tile.
* Codebook mul1 (cb 2, multiplier 0x83DCD12D): `x = state*mult (u32)`, `s = sum of 4 bytes of x`, value `= fma(fp16(1024+s), fp16(0x1eee), fp16(0xc931))` rounded once to fp16 (unit-variance, min -3.45 / max +3.35). mcg (cb 1, 0xCBAC1FED) and 3inst (cb 0) are implemented from codebook.cuh (`(x & 0x8fff8fff) ^ 0x3b603b60`, sum of the two fp16 halves); **mcg/3inst are NOT validated on real data** because every tensor in this checkpoint is mul1 (all 17 `mul1` markers I pulled are the i32 `-2082680531` = `0x83DCD12D` = 2212286765). The marker's value is not used by the kernel (the codebook is a compile-time id); `codebook_from_multiplier` just maps it to the id and rejects anything else.
* Hadamard = Sylvester (natural order) H128 / sqrt(128) on independent 128-element blocks, symmetric and involutory (the 4-element in-thread butterfly + xor-shuffle in hadamard_inner.cuh is exactly this).
* `y = svh * H_out( H_in(x * suh) @ W_inner )`, i.e. `W_eff[in,out] = diag(suh) . blockdiag(H128) . W_inner . blockdiag(H128) . diag(svh)` and `y = x @ W_eff`. This is what `dequant_exl3` returns and what `exl3_linear_ref` computes stage by stage (the two agree to 5e-7 relative on every tensor).

## Results (random x, shape [8, in], seed fixed; W error is ||W_exl3 - W_base|| / ||W_base|| with W_base transposed to [in, out])

| tensor | bpw | in x out | W rel err | y rel err | y rel err, fp16-staged emulation |
|---|---|---|---|---|---|
| layers.0.attn.wq_a | 6 | 5120 x 1280 | 2.03 % | 2.04 % | 2.04 % |
| layers.0.attn.wo_a.slice.3 vs base rows 3072:4096 | 6 | 4096 x 1024 | 2.02 % | 2.00 % | 2.00 % |
| layers.0.ffn.shared_experts.w1 | 6 | 5120 x 2304 | 2.10 % | 2.12 % | 2.12 % |
| layers.0.ffn.shared_experts.w2 | 6 | 2304 x 5120 | 2.09 % | 2.06 % | 2.06 % |
| layers.0.ffn.experts.0.w1 (vs MXFP4) | 4 | 5120 x 2304 | 7.94 % | 7.93 % | 7.93 % |
| layers.12.ffn.experts.0.w1 (vs MXFP4) | 3 | 5120 x 2304 | 14.42 % | 14.33 % | 14.33 % |
| layers.2.attn.indexer.wk (vs BF16) | 8 | 512 x 128 | 0.65 % | 0.70 % | 0.70 % |
| layers.1.engram.wkv | 6 | 6144 x 25600 | 1.86 % | 1.87 % | 1.87 % |
| layers.2.attn.compressor.wkv (vs BF16) | 6 | 5120 x 512 | 2.00 % | 2.06 % | 2.07 % |
| layers.2.attn.compressor.wgate (vs BF16) | 6 | 5120 x 512 | 2.00 % | 2.07 % | 2.07 % |

(The last column rounds A_had, the accumulator and the output to fp16 at the points the kernels do; it differs from the fp32 path by <= 1e-4 relative, so fp16 kernel rounding is a non-issue. Engram took 4 s and 0.7 GB RSS thanks to row-block streaming.)

wo_a: all 8 slices checked against all 8 row groups of the grouped base weight `[8192, 4096]` (FP8 + `[256,128]` E8M0 scales):

| slice G | trellis shape (K=4096, N=1024, 6 bpw) | W rel err vs base rows G*1024:(G+1)*1024 | best of the 8 row groups | next best group |
|---|---|---|---|---|
| 0 | [256, 64, 96] | 1.99 % | G | >= 127 % |
| 1 | same | 2.04 % | G | >= 136 % |
| 2 | same | 2.05 % | G | >= 136 % |
| 3 | same | 2.02 % | G | >= 143 % |
| 4 | same | 2.06 % | G | >= 140 % |
| 5 | same | 2.13 % | G | >= 133 % |
| 6 | same | 2.03 % | G | >= 138 % |
| 7 | same | 2.02 % | G | >= 135 % |

So slice g == base rows g*1024:(g+1)*1024 (base layout `[out = 8 groups x 1024, in = 4096]`), stored with in = 4096 (suh `[4096]`), out = 1024 (svh `[1024]`).

MXFP4 base decoding (not part of EXL3 but needed for tests 4): `I8 [2304, 2560]` = two E2M1 per byte (table 0,.5,1,1.5,2,3,4,6, sign = bit 3), E8M0 scale `[2304, 160]` per 32 inputs, `2^(b-127)`. **Low nibble = first (even) element**: 7.94 % vs 141 % for the other order, so that is settled empirically.

## Answers to (a)-(e)

**(a) Layout.** W is stored `[in, out]`: trellis dim 0 = K/16 (input), dim 1 = N/16 (output); `y = x @ W`. The base checkpoint is `[out, in]`; the comparison transposes it.

**(b) suh / svh.** `suh [K]` (fp16) multiplies the INPUT features element-wise BEFORE the input Hadamard (`had128(x * suh)`); `svh [N]` multiplies the OUTPUT features AFTER the output Hadamard. Putting them anywhere else is wrong (W_rel on wq_a):

| variant | W rel err |
|---|---|
| kernel order `diag(suh) H W_inner H diag(svh)` | 2.03 % |
| suh after the input Hadamard | 140.9 % |
| svh before the output Hadamard | 140.9 % |
| no svh | 142.7 % |
| no Hadamards | 141.5 % |
| no suh | 4168 % |
| 16x16 tile (k,n) transposed | 137.0 % |

**(c) Zero `svh` on trailing whole 128-column blocks gives exact zeros, independent of the trellis there: CONFIRMED.** wq_a (N=1280), svh[-256:] = 0: columns 0..1023 bit-identical to the unmodified result; columns 1024..1279 are exactly 0.0, and stay exactly 0.0 (and the real columns bit-identical) when the trellis in those 16 tile-columns is replaced by zero words or by two different random int16 garbage fills (also with the fp16-staged emulation). Reason: the output Hadamard only mixes within a 128-block and `0 * finite = 0`; the codebook is finite for all 65536 states (checked via the LUT; the plugin's comment says the same). Negative control: a pad that straddles a block (svh zero on the last 192 columns) with garbage trellis in the pad corrupts the real columns 1024..1087 sharing the block (rel change 161 %), while whole blocks before it are untouched. Note a zero-WORD trellis is not a zero weight: mul1 `LUT[0] = -3.453`, mcg `LUT[0] = +1.84`.

**(d) Zero `suh` on trailing whole 128-row blocks leaves the output unchanged: CONFIRMED (bit-exact in the reference).** wq_a K = 5120 padded by 256 rows (16 trellis tile-rows) with suh = 0: output bit-identical (max |diff| = 0.0) for pad trellis rows = zero words, = random garbage, and for pad activations = 0 or finite junk (randn*1000). It is the trellis rows that do not matter (A_had is exactly 0 in the pad columns), and the matching trellis rows need not be zero. On a GPU the sum order of the K loop may differ, so expect fp32-rounding-level, not bitwise, equality there. **Gotcha: the pad activations must be finite**: `NaN * 0 = NaN`, one NaN in a padded x row turns that whole token row of the output into NaN (other rows stay clean). Negative control: padding that is not a whole 128-block (suh = 0 on the last 64 rows of a real block) makes the output depend on the pad trellis content (rel difference zero-words vs garbage = 41 %).

**(e) Tensor-parallel splits.** Real tensors `shared_experts.w1` (5120 -> 2304) and `.w2` (2304 -> 5120).
* Column-parallel (split OUTPUT, narrow trellis dim 1 by tile index + svh, full suh): at 128 multiples (2 ranks @1152, 3 ranks @768, uneven @640/1280) the concatenated rank outputs are bit-identical to the full result. Row-parallel (split INPUT, narrow trellis dim 0 + suh + x, full svh on every rank): the sum of the rank partials equals the full result to 3e-7 relative (fp32 reorder only; svh being linear it is correct to apply it per rank before the all-reduce).
* **Non-128 split at 576 of 2304 breaks it, for both directions.** There is no valid way to compute a 576-wide shard (4.5 blocks); the best naive emulation (zero-pad the rank-local tail to a full block, drop pad outputs) gives: column split, columns [0,512) exact (whole blocks), straddled block [512,576) rel err 295 %, rank 1 columns [576,2304) (blocks re-aligned to 576) rel err 179 %, overall 161 %; row split: sum of partials rel err 158 %.
* **What works instead for TP4 on a 2304 dim:** pad the tail by whole blocks to 2560 = 4 x 640 (5 blocks per rank). Test: w1 output padded 2304 -> 2560 (svh = 0 on the 256 pad columns, garbage trellis there), split @640 over 4 ranks: real columns bit-identical to the unpadded full result, pad columns exactly 0. w2 input padded 2304 -> 2560 (suh = 0, garbage trellis rows, junk finite x), 4-rank row split @640: sum of partials 3.1e-7 rel err vs unpadded. The chained padded w1 -> w2 matches the unpadded chain to 3.0e-7 (the zero pad columns coming out of w1 feed straight into w2's zero-suh rows). The last rank then holds 384 real + 256 pad; other ranks are full.

## Gotchas

1. The three kernel headers are missing from the scratchpad plugin copy (see Source of truth); everything here rests on upstream + the numerical agreement.
2. Plugin source reading (not run on a GPU): `exl3_linear` checks `k % 32 == 0` and `exl3_had_in` checks `k % 128 == 0`, but nothing checks `n_total % 128` for non-vocab layers (`create_weights` only requires %16 and %128 for the input slice). The gemm grid is `n / BN_` with `BN_ = 128`, so a trailing partial 128-column block would be silently skipped and (torch.empty output) left uninitialized. A column-parallel rank width that is not a multiple of 128 (e.g. 576) must therefore be rejected or padded, not just passed through. The MoE gemm does `TORCH_CHECK(group_n[g] % 128 == 0)`.
3. Zero trellis words decode to non-zero weights (see (c)); only `suh = 0` (input pad) or `svh = 0` (output pad) on whole 128-blocks neutralizes padding.
4. Pad activations must be finite (0 * inf/NaN = NaN).
5. mcg / 3inst codebooks are implemented from source but unvalidated on real data (checkpoint is all mul1).
6. bf16 activations: kernels convert x to fp16 for the Hadamard/mma; the reference does everything in fp32 unless `emulate_fp16=True` (difference <= 1e-4 relative).
7. Reddie's system python has no numpy/torch/safetensors; the pull helper is pure python. A Tailscale SSH "additional check" prompt appeared once mid-session and cleared by itself; later calls went through.
