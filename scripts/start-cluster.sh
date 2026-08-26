#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

for host in "${SPARK_1_HOST}" "${SPARK_2_HOST}"; do
  ssh "${host}" "test -f '${REMOTE_INSTALL_DIR}/.env'"
  ssh "${host}" "test -s '${REMOTE_INSTALL_DIR}/.sglang-api-key'"
  ssh "${host}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/verify-model.py '${MODEL_DIR}'"
  ssh "${host}" "docker image inspect '${IMAGE_REF}' >/dev/null"
done

ssh "${SPARK_1_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/stop-node.sh"
ssh "${SPARK_2_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/stop-node.sh"

# Start rank 1 first so it is waiting when rank 0 initializes the API server.
ssh "${SPARK_2_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/start-node.sh 1"
ssh "${SPARK_1_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/start-node.sh 0"

echo "waiting for Qwen SGLang API"
for attempt in $(seq 1 120); do
  if ssh "${SPARK_1_HOST}" "cd '${REMOTE_INSTALL_DIR}' && ./scripts/probe-api.py" \
    | grep -q 'qwen3.8-flash-next'; then
    echo "API ready after attempt ${attempt}"
    exit 0
  fi

  for host in "${SPARK_1_HOST}" "${SPARK_2_HOST}"; do
    if ! ssh "${host}" "docker ps --format '{{.Names}}'" \
      | grep -qx "${CONTAINER_NAME}"; then
      echo "${host} exited before readiness" >&2
      ssh "${host}" "docker logs --tail 200 '${CONTAINER_NAME}'" || true
      exit 1
    fi
  done
  sleep 15
done

echo "timed out waiting for Qwen SGLang API" >&2
exit 1
