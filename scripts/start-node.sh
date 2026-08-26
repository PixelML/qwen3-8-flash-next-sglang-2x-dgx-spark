#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[01]$ ]]; then
  echo "usage: $0 <node-rank: 0|1>" >&2
  exit 2
fi

NODE_RANK="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || {
  echo "model checkpoint missing: ${MODEL_DIR}" >&2
  exit 1
}
[[ -s "${SECRET_FILE}" ]] || {
  echo "API secret missing or empty: ${SECRET_FILE}" >&2
  exit 1
}
chmod 600 "${SECRET_FILE}"

"${SCRIPT_DIR}/apply-sm121-qsa-patch.sh"
docker image inspect "${IMAGE_REF}" >/dev/null

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm --force "${CONTAINER_NAME}" >/dev/null
fi

mkdir -p \
  "${REPO_ROOT}/cache/huggingface" \
  "${REPO_ROOT}/cache/torch" \
  "${REPO_ROOT}/cache/triton"

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --gpus all \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --cap-add IPC_LOCK \
  --device /dev/infiniband \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  --env NODE_RANK="${NODE_RANK}" \
  --env DIST_INIT_ADDR="${RANK0_ADDR}:${DIST_PORT}" \
  --env API_PORT="${API_PORT}" \
  --env CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  --env MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
  --env MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  --env SGLANG_API_KEY_FILE=/run/secrets/sglang-api-key \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  --env GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  --env NCCL_NET=IB \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_IB_HCA="${NCCL_IB_HCA}" \
  --env NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}" \
  --env NCCL_IB_ADDR_FAMILY=AF_INET \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_CROSS_NIC=1 \
  --env NCCL_CUMEM_ENABLE=0 \
  --env NCCL_NVLS_ENABLE=0 \
  --env NCCL_IGNORE_CPU_AFFINITY=1 \
  --env NCCL_DEBUG=WARN \
  --env TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  --env TORCH_NCCL_ENABLE_MONITORING=1 \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/torch \
  --env TRITON_CACHE_DIR=/cache/triton \
  --mount type=bind,src="${MODEL_DIR}",dst=/models/qwen38-flash-next,readonly \
  --mount type=bind,src="${SECRET_FILE}",dst=/run/secrets/sglang-api-key,readonly \
  --mount type=bind,src="${SCRIPT_DIR}/launch_secure.py",dst=/opt/qwen/launch_secure.py,readonly \
  --mount type=bind,src="${SCRIPT_DIR}/redact_stream.py",dst=/opt/qwen/redact_stream.py,readonly \
  --mount type=bind,src="${REPO_ROOT}/build/qwen_sparse_attn_backend.py",dst=/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py,readonly \
  --mount type=bind,src="${REPO_ROOT}/cache/huggingface",dst=/root/.cache/huggingface \
  --mount type=bind,src="${REPO_ROOT}/cache/torch",dst=/cache/torch \
  --mount type=bind,src="${REPO_ROOT}/cache/triton",dst=/cache/triton \
  --entrypoint bash \
  "${IMAGE_REF}" \
  -lc 'set -o pipefail; python3 -u /opt/qwen/launch_secure.py \
    --model-path /models/qwen38-flash-next \
    --served-model-name qwen3.8-flash-next \
    --tp 2 \
    --nnodes 2 \
    --node-rank "${NODE_RANK}" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --linear-attn-prefill-backend triton \
    --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 \
    --mamba-scheduler-strategy extra_buffer \
    --mamba-track-interval 64 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --allow-auto-truncate \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --sampling-defaults model \
    --watchdog-timeout 1200 \
    --enable-metrics \
    --host 0.0.0.0 \
    --port "${API_PORT}" 2>&1 | python3 -u /opt/qwen/redact_stream.py'

echo "started ${CONTAINER_NAME} rank=${NODE_RANK}"
