#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/.claude-config"
SETTINGS_FILE="${CONFIG_DIR}/settings.json"
ENV_FILE="${CONFIG_DIR}/zai.env"

if [[ -z "${ZAI_API_KEY:-}" ]]; then
  echo "Error: ZAI_API_KEY is not set."
  echo "Run: ZAI_API_KEY=... ${BASH_SOURCE[0]}"
  exit 1
fi

BASE_URL="${ZAI_BASE_URL:-https://api.z.ai/api/anthropic}"
TIMEOUT_MS="${ZAI_API_TIMEOUT_MS:-3000000}"

mkdir -p "${CONFIG_DIR}"

cat > "${SETTINGS_FILE}" <<JSON
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "${ZAI_API_KEY}",
    "ANTHROPIC_BASE_URL": "${BASE_URL}",
    "API_TIMEOUT_MS": "${TIMEOUT_MS}"
  }
}
JSON

cat > "${ENV_FILE}" <<ENVVARS
ANTHROPIC_AUTH_TOKEN=${ZAI_API_KEY}
ANTHROPIC_BASE_URL=${BASE_URL}
API_TIMEOUT_MS=${TIMEOUT_MS}
ENVVARS

echo "Wrote ${SETTINGS_FILE}"
echo "Wrote ${ENV_FILE}"
echo "Next:"
echo "  ${SCRIPT_DIR}/claude_zai.sh"
