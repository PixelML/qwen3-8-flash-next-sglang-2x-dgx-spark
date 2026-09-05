#!/usr/bin/env bash
# Boot a mixed-precision ModelOpt NVFP4 checkpoint on 2x DGX Spark under upstream
# vLLM, TP=2 over the RoCE link. No Ray: native --nnodes/--node-rank.
#
# The follower (rank != 0) MUST run --headless, or its EngineCore aborts with
# "AssertionError: collective_rpc should not be called on follower node".
#
# GB10 memory warning: the ~120 GiB is UNIFIED (host and device share one pool),
# so --gpu-memory-utilization does not mean what it means on a discrete GPU.
# 0.85 wedged both nodes past weight load on 2026-09-05; start at 0.70-0.75.
#
# Set MASTER_ADDR / WORKER_HOST_IP to the two nodes' addresses on the RoCE link.
# Usage: boot-nvfp4-vllm.sh <node-rank> <model-dir> <tag> [extra vllm args...]
set -euo pipefail
RANK="$1"; MODEL_DIR="$2"; TAG="$3"; shift 3
EXTRA=$(printf "%q " "$@")
IMAGE="vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174"
MASTER_ADDR="${MASTER_ADDR:?set the rank-0 node IP on the RoCE link}"
MASTER_PORT="${MASTER_PORT:-25100}"
if [ "$RANK" = "0" ]; then HOST_IP="$MASTER_ADDR"; HEADLESS_ARG=""; else HOST_IP="${WORKER_HOST_IP:?set the rank-1 node IP on the RoCE link}"; HEADLESS_ARG="--headless"; fi
NAME="nvfp4-vllm-r${RANK}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$HOME/nvfp4-vllm-logs" "$HOME/.cache/nvfp4-vllm-tmp"
docker run -d --name "$NAME" \
  --network host --ipc host --shm-size 64gb \
  --gpus all --device /dev/infiniband:/dev/infiniband \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR":/model:ro \
  -v "$HOME/.cache/huggingface":/cache/huggingface \
  -v "$HOME/.cache/nvfp4-vllm-tmp":/tmp \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e TP_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TRITON_CACHE_DIR=/cache/huggingface/triton-cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e HEADLESS_ARG="$HEADLESS_ARG" \
  --entrypoint bash \
  "$IMAGE" -lc "exec vllm serve /model \
    --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8890 \
    --trust-remote-code \
    --quantization modelopt \
    --tensor-parallel-size 2 \
    --nnodes 2 --node-rank $RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    $HEADLESS_ARG $EXTRA 2>&1"
echo "started $NAME ($TAG)"
