#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_PARENT="$(dirname "${MODEL_DIR}")"
MODEL_NAME="$(basename "${MODEL_DIR}")"
mkdir -p "${MODEL_PARENT}"
MODEL_PARENT="$(cd "${MODEL_PARENT}" && pwd -P)"

docker pull "${IMAGE_REF}"
docker run --rm \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env HF_HOME=/models/.hf-cache \
  --volume "${MODEL_PARENT}:/models" \
  "${IMAGE_REF}" \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='/models/${MODEL_NAME}', max_workers=2)"

python3 "${SCRIPT_DIR}/verify-model.py" "${MODEL_DIR}"
