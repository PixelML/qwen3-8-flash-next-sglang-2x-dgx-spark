#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
  --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
"${SCRIPT_DIR}/probe-api.py" || true
