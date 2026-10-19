# Decision: DeepSeek-V4.1-Flash-1017 on two DGX Sparks (TP2), TensorFold native engine (Phase A, 2026-10-18)

## Engine
- **Engine**: [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) branch `deepseek-v41-zig` at `3ff3b2a` (the recipe's engine line; the recipe's served commit `65f295ec` is not in the public history, `ce20a54` is its code). `3ff3b2a` already contains `47a8910` (Triton variant fallback), `b8ae0c7` (indexer alignment), `6a1af41` (RDMA GID index `TF_RDMA_GID_INDEX` and MTU negotiation) and `9a0c014` (loader reads each layer's experts whole): so TP4 patches 0001 to 0003 are **not needed**.
- **Three patches on top** ([patches/](patches/)): `0001` = bertholomus's `5940208` (`tf-dsv41-rank-cache`, the no-GPU rank-cache writer, plus `node_inputs.sh`), cherry-picked with README/kit-tool conflicts resolved in favour of the branch; `0002` = our TP4 patch 0004 (random seed per request, top_k handling; applies cleanly); `0003` = the rank-cache tools take 1 or 2 ranks only (the parity split of `5940208` does not exist on this branch; the first build failed on it).
- **Why not our TP4 engine with `TF_TP_WORLD=2`**: it forks before `ce20a54`, lacks `6a1af41` and `9a0c014`, and its TP2 path was not exercised on this model; the TP2 recipe's line is the one bertholomus measured at 89.8 to 143 tok/s on the Mia 2.9 pack. The TP4 build still has a TP2 code path and could be a fallback if the new engine misbehaves.
- Built in `nvcr.io/nvidia/pytorch:26.07-py3` with Zig 0.17.0, `-Dsm=121`, rc 0, about 2 minutes, **no GPU attached** ([launch/build.sh](launch/build.sh)). Binaries: `/var/tmp/tf1017-tp2/engine-bin/` on Reddie and Bluey.

## Kit
- Recipe `kit/` v0.6.1 (tag, `MANIFEST` 223 files), copied to `/var/tmp/tf1017-tp2/kit` on both nodes. Verified with its own `verify_kit.sh`: **222 of 223** match. The one difference is `vision/gate_bias_vl.safetensors`, which is the 1017 weights' own bias (sha `4745da4b...`, same as the TP4 lane). The four RoPE tables and `engram.json` are byte-identical to the TP4 kit's (their sha matches the recipe's `rope.json`), so they were copied, not recomputed.
- **Kernel coverage** ([results/k-coverage-2.9bpw.txt](results/k-coverage-2.9bpw.txt)): the pack's trellis bit widths are K = 2 (5,760 tensors: layers 18-22 routed experts), 3 (40,320 routed experts), 4 (1,209: shared experts on layers 11-28, MTP experts, engram wkv), 5 (558: attention, shared experts), 6 (49: head, layer-0 `wq_a`/`wkv`, **all MTP/DSpark dense parts**, `mtp.0.main_proj`) and 8 (4: indexer `wk`). The kit's `linear` and `linear_grouped` cubins contain kernels for K2 = 2..16 (K = 1 to 8 bits), so the six-bit MTP dense tensors are covered; the expert cubins have the `(2,10)`, `(2,16)` and `(8,8)` half-bit ranges (3-bit and 2-bit routed experts; 4-bit MTP experts). Nothing needs `build_kit.sh`. This is a static check; the first GPU confirmation is the load in Phase B.
- **Loader plan**: `tf-dsv41-rank-cache --index-only` accepts the pack for both ranks (no missing or mis-shaped tensor): each rank cache is **106.19 GB (98.9 GiB)**.

## Nodes
- **Rank 0 = Reddie** (192.168.192.2; keeps 100.113.138.96:8000 for the fleet), **rank 1 = Bluey** (192.168.192.1). Reasons: the endpoint IP stays; Bluey holds the pack and has 1.5 TiB free; Reddie has the room for its 98.9 GiB cache (187 GiB free before the copy, about 88 GiB after).
- Memory: 98.9 GiB weights per node out of about 117 GiB. The recipe's measured floor at `--context 1048576 --parallel 4` is about 5 GiB left on the head. Reddie is tight and also the fleet head; if rank 0 runs short, swap the roles (Bluey rank 0, endpoint moves) or lower `CONTEXT`.
- Spark4 (230 GiB free) is the alternative second node if Bluey is wanted for something else; it would need the pack, which would have to be copied from Bluey first (about 17 min).

## Served settings (config.tp2.env)
`--context 1048576`, `--parallel 4` (engine default `TF_DS_STREAMS=4`, 16 round rows), the engine's pool = 1,048,576 positions shared by the streams, both RoCE rails (`rocep1s0f0,roceP2p1s0f0`), `TF_RDMA_GID_INDEX=3` + `NCCL_IB_GID_INDEX=3`, `TF_DS_REPLAY_FLOOR=1`, patch sampling defaults (temperature 1.0, top_p 1.0, top_k 20, `TF_SEED_RANDOM=1`, `TF_TOP_K_MAX=56`), thinking off by default, names `deepseek-v4.1-flash` and `glm-5.3-flash`, port 8000 on 0.0.0.0. `config.tp2-s8.env` is an 8-stream experiment, only if memory shows headroom.

## Expected numbers (to be replaced by Phase B measurements)
The recipe measured on the Mia 2.9 pack: decode 89.8 to 143.2 tok/s single stream (by prompt type), 83 aggregate at 2 streams, 119 to 142 at 4, prompt 2,354 to 2,495 tok/s from 8K to 128K. The LibertAI pack has the same layout; the 6-bit MTP dense parts make the drafter slightly heavier, so expect equal or a little lower.

## Things that need a decision
1. **The live fleet lane is the uncensored (Keys overlay) TP4 build** (`start_fleet_rail2_p4_ablit.sh`, `cache-ablit`). The TP2 lane built here serves the plain LibertAI pack, so while TP2 holds `:8000` the fleet gets the non-uncensored model, with 4 streams instead of 16. Keep the Phase B window short, or ask for an overlaid TP2 cache first (not built; the overlay recipe for the 2.9 bpw EXL3 tensors is unverified).
2. The TP2 lane replaces all four nodes' GPU use, so Asusi and Spark4 idle during Phase B. 
3. Phase B stops the live endpoint (full outage while each lane loads, see RUNBOOK_PHASE_B.md).
