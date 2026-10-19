# Status: DeepSeek-V4.1-Flash-1017 on TensorFold, TP2 (updated 2026-10-19)

Served and measured on 2026-10-19 00:01 to 00:31 UTC (first boot, no restarts). The fleet endpoint was then put back on the four-node lane.

| check | result |
|---|---|
| Engine build (Zig 0.17.0, sm_121, `nvcr.io/nvidia/pytorch:26.07-py3`, no GPU) | rc 0, about 2 min ([logs/build.log](logs/build.log)) |
| Kit manifest | 222 of 223; the bias file differs, as in the 1017 TP4 lane ([logs/kit_verify.log](logs/kit_verify.log)) |
| Kernel coverage of the pack | K = 2, 3, 4, 5, 6, 8 all present in the kit's cubins ([results/k-coverage-2.9bpw.txt](results/k-coverage-2.9bpw.txt)); confirmed by the load |
| Loader plan | accepted for both ranks, 106.19 GB per rank |
| Rank caches and token map | see [logs/inputs.log](logs/inputs.log) |
| Boot | loaded in 27.1 s (rank 1: 26.6 s); model 105.04 GiB per node; 4 streams, 1,048,576-token window; both RoCE rails up, `gid_index=3` ([logs/boot/](logs/boot/)) |
| Gates | `sanity.py` 6 of 6 PASS, `fleet_check.py` 14 PASS 0 FAIL ([logs/sanity.log](logs/sanity.log), [logs/fleet_check_tailnet.log](logs/fleet_check_tailnet.log)); concurrent == solo 8 of 8; sampling checks all pass |
| Speed ([results/tp2a](results/tp2a/)) | C1 73.1, C2 106.0, C3 123.7, C4 135.8 tok/s aggregate; cold prefill 1,922 to 2,200 tok/s from 2.9K to 93K tokens, 2,044 at 187K |
| Needles ([logs/long_check_big.log](logs/long_check_big.log)) | found at 131,073; 300,000 (153 s); 500,000 (275 s); 1,000,000 (642 s, 1,559 tok/s) |
| 69-scenario eval ([results/eval69](results/eval69/)) | quality 97.1 (134/138: 66 pass, 2 partial, 1 fail), responsiveness 96.5, deployability 96.9, 71.0 tok/s decode |
| Memory while serving | 6.7 GiB available on rank 0, 8.2 GiB on rank 1 (no room for a larger stream count; the 8-stream config was not tried) |
| Restore of the four-node lane | answered 2 min after start; `fleet_check.py` 14 PASS ([logs/fleet_check_tp4_restored.log](logs/fleet_check_tp4_restored.log)) |
