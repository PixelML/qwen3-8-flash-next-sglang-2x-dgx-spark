#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ssh "${SPARK_1_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/stop-node.sh"
ssh "${SPARK_2_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/stop-node.sh"
