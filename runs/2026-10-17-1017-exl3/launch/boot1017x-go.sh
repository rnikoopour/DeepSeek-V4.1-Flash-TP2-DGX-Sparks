#!/bin/bash
# boot1017x: the full-EXL3 DeepSeek-V4.1-Flash-1017 lane (LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) at TP4 on the 4 Sparks.
# Same serving config and levers as boot1017g (the native 1017 lane): CUDA graphs FULL_AND_PIECEWISE, DSpark k=5, vision + tools on,
# thinking off, 300K, gmu 0.80, NCCL 8 channels, Engram fast staging + 128 read threads, b12x RoCE all-reduce/all-gather, indexer
# prefill TP-split, expandable_segments:False, served as deepseek-v4.1-flash AND glm-5.3-flash. Differences:
#   * model dir = the EXL3 checkpoint (attention, shared experts, indexer, compressor, Engram projection, head, MTP/DSpark all EXL3);
#   * patch set dsv41-exl3-1017x (patch/exl3-1017-full + the sr4 EXL3 set), image vllm-dsv41:exl3b-roce (cuda-exl3 plugin);
#   * the Engram tables are read from the base 1017 checkpoint (/engram-base) and the node-local copies (/engram-local).
export EXP_NAME=exl3-1017x IMAGE=vllm-dsv41:exl3b-roce PATCH_NAME=dsv41-exl3-1017x ENGRAM_LOCAL=1 GMU=0.80 MAXLEN=300000 SEQS=8 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0
export MODEL_DIR=DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw ENGRAM_BASE_NAME=DeepSeek-V4.1-Flash-1017 ENGRAM_LOCAL_HOST=/var/tmp/engram-local/DeepSeek-V4.1-Flash-1017 SERVED_NAMES="deepseek-v4.1-flash glm-5.3-flash"
export ENGRAM_THREADS=128
export NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton"
export VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1'
export NCCL_EXTRA="$NCCL_EXTRA -e NCCL_MAX_NCHANNELS=8 -e DSV41_ENGRAM_FAST=1"
export NCCL_EXTRA="$NCCL_EXTRA -e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB -e VLLM_ROCE_ALLGATHER_ENABLE=1 -e B12X_ROCE_HCA=rocep1s0f0 -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=300000000 -e B12X_ROCE_CACHE_DIR=/opt/b12x-roce/cache"
export NCCL_EXTRA="$NCCL_EXTRA -e DSV41_INDEXER_TP_SPLIT=1"
export NCCL_EXTRA="$NCCL_EXTRA -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
# EXL3 dense GEMMs (now including the shared experts) share one process-wide activation / split-k workspace in the plugin, so a
# shared-expert GEMM on an aux stream next to the routed experts would race on it: keep the shared experts on the main stream.
export NCCL_EXTRA="$NCCL_EXTRA -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1"
echo "boot1017x go $(date -u +%FT%TZ) model=$MODEL_DIR names=[$SERVED_NAMES] image=$IMAGE patch=$PATCH_NAME threads=${ENGRAM_THREADS:-32} extra=[$NCCL_EXTRA] gmu=$GMU maxlen=$MAXLEN seqs=$SEQS spec=$SPEC k=$SPEC_K eager=$EAGER cg=$CUDAGRAPH_MODE"
bash $HOME/boot_dsv41_1017x.sh
echo "boot_dsv41_1017x exit=$? $(date -u +%FT%TZ)"
