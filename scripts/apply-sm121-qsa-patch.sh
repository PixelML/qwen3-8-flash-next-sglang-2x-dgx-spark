#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SOURCE_PATH="/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
PATCH_FILE="${REPO_ROOT}/patches/sglang-qsa-sm121-xqa.patch"
OUTPUT_DIR="${REPO_ROOT}/build"
OUTPUT_FILE="${OUTPUT_DIR}/qwen_sparse_attn_backend.py"
TEMP_FILE="${OUTPUT_FILE}.tmp"

mkdir -p "${OUTPUT_DIR}"
docker image inspect "${IMAGE_REF}" >/dev/null
docker run --rm --entrypoint cat "${IMAGE_REF}" "${SOURCE_PATH}" > "${TEMP_FILE}"
patch --quiet --unified "${TEMP_FILE}" "${PATCH_FILE}"
mv "${TEMP_FILE}" "${OUTPUT_FILE}"
python3 -m py_compile "${OUTPUT_FILE}"
echo "prepared ${OUTPUT_FILE}"
