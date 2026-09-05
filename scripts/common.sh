#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "configuration missing: ${ENV_FILE}; copy .env.example to .env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

IMAGE_REF="lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1"
MODEL_REPO="${MODEL_REPO:-PixelML/Qwen3.8-Flash-Next-NVFP4-Dual-DGX-Spark}"
MODEL_REVISION="${MODEL_REVISION:-b80180e371f13348ec49641a6e66999e7854b179}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-sglang}"
# ModelOpt scheme of the checkpoint in MODEL_DIR. The verified lane weights
# are uniform NVFP4 (modelopt_fp4). NVIDIA's own Qwen3.8-Flash-Next-NVFP4 is
# a MIXED_PRECISION ModelOpt checkpoint and needs modelopt_mixed instead.
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
# See scripts/start-node.sh; 1 preserves the published behaviour.
STRICT_MEM_CHECK_DURING_IDLE="${STRICT_MEM_CHECK_DURING_IDLE:-1}"
SECRET_FILE="${REPO_ROOT}/.sglang-api-key"

required=(
  MODEL_DIR RANK0_ADDR DIST_PORT NCCL_SOCKET_IFNAME NCCL_IB_HCA
  NCCL_IB_GID_INDEX API_PORT CONTEXT_LENGTH MAX_RUNNING_REQUESTS
  MEM_FRACTION_STATIC
)

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "required configuration is empty: ${name}" >&2
    exit 1
  fi
done
