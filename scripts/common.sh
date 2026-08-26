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
MODEL_REPO="RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
CONTAINER_NAME="qwen38-flash-next-sglang"
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
