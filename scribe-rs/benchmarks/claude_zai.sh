#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/.claude-config"
ENV_FILE="${CONFIG_DIR}/zai.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: ${ENV_FILE} not found. Run ${SCRIPT_DIR}/setup_zai_claude_config.sh first."
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

export CLAUDE_CONFIG_DIR="${CONFIG_DIR}"
unset ANTHROPIC_API_KEY

exec claude "$@"
