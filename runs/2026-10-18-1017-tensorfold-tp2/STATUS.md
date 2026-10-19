# Status: DeepSeek-V4.1-Flash-1017 on TensorFold, TP2 (updated 2026-10-18)

Prepared, not yet served. Nothing has run on a GPU.

| check | result |
|---|---|
| Engine build (Zig 0.17.0, sm_121, `nvcr.io/nvidia/pytorch:26.07-py3`, no GPU) | rc 0, about 2 min ([logs/build.log](logs/build.log)) |
| Kit manifest | 222 of 223; the bias file differs, as in the 1017 TP4 lane ([logs/kit_verify.log](logs/kit_verify.log)) |
| Kernel coverage of the pack | K = 2, 3, 4, 5, 6, 8 all present in the kit's cubins ([results/k-coverage-2.9bpw.txt](results/k-coverage-2.9bpw.txt)); static check, GPU confirmation pending |
| Loader plan | accepted for both ranks, 106.19 GB per rank |
| Rank caches and token map | see [logs/inputs.log](logs/inputs.log) |
| Boot, gates, bench, eval | pending |
