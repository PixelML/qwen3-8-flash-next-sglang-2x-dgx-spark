#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SRT_PREFIX="/sgl-workspace/sglang/python/sglang/srt"
PATCH_DIR="${REPO_ROOT}/patches"
OUTPUT_DIR="${REPO_ROOT}/build"

# Each entry: <output basename>|<patch file>|<in-image source path, '-' for a verbatim copy>
ENTRIES=(
  "qwen_sparse_attn_backend.py|${PATCH_DIR}/sglang-qsa-sm121-fallback.patch|${SRT_PREFIX}/layers/attention/qwen_sparse_attn_backend.py"
  "schedule_batch.py|${PATCH_DIR}/sglang-token0-guard-schedule_batch.patch|${SRT_PREFIX}/managers/schedule_batch.py"
  "batch_result_processor.py|${PATCH_DIR}/sglang-token0-guard-batch-result-processor.patch|${SRT_PREFIX}/managers/scheduler_components/batch_result_processor.py"
  "scheduler.py|${PATCH_DIR}/sglang-token0-guard-scheduler.patch|${SRT_PREFIX}/managers/scheduler.py"
  "sm121_varlen.py|${PATCH_DIR}/sm121_varlen.py|-"
)

mkdir -p "${OUTPUT_DIR}"

if ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
  echo "docker image missing: ${IMAGE_REF}; run docker pull first" >&2
  exit 1
fi

apply_entry() {
  local output_name="$1" patch_file="$2" image_path="$3"
  local output_file="${OUTPUT_DIR}/${output_name}"
  local temp_file="${output_file}.tmp"

  if [[ ! -f "${patch_file}" ]]; then
    echo "patch file missing: ${patch_file}" >&2
    exit 1
  fi

  if [[ "${image_path}" == "-" ]]; then
    echo "staging ${patch_file}"
    cp -- "${patch_file}" "${temp_file}"
  else
    echo "extracting ${image_path}"
    if ! docker run --rm --entrypoint cat "${IMAGE_REF}" "${image_path}" > "${temp_file}"; then
      echo "failed to extract ${image_path} from ${IMAGE_REF}" >&2
      rm -f "${temp_file}"
      exit 1
    fi
    echo "patching with ${patch_file}"
    if ! patch --quiet --unified "${temp_file}" "${patch_file}"; then
      echo "patch failed: ${patch_file} against ${image_path}" >&2
      rm -f "${temp_file}"
      exit 1
    fi
  fi

  if ! python3 -m py_compile "${temp_file}"; then
    echo "py_compile failed: ${output_name}" >&2
    rm -f "${temp_file}"
    exit 1
  fi
  mv -- "${temp_file}" "${output_file}"
  echo "prepared ${output_file}"
}

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r output_name patch_file image_path <<< "${entry}"
  apply_entry "${output_name}" "${patch_file}" "${image_path}"
done

echo "all SM121/token0-guard patches staged in ${OUTPUT_DIR}"
