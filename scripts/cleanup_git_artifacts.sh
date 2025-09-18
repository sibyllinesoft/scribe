#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

paths=(
  "scribe-rs/target.old"
  "scribe-rs/target_new"
  "scribe-rs/target"
  "scribe-rs/test-artifacts"
  "scribe-rs/test-ci.yml"
  "scribe-rs/test-environment.yml"
  "scribe-rs/.actrc"
  "scribe-rs/.secrets"
  "artifacts"
  "test_progress.html"
  "tests/webui/tree-building.test.js"
  ".actrc"
  ".secrets"
  ".github/actions"
)

for path in "${paths[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Cleaning $path"
    chmod -R u+rwX "$path" 2>/dev/null || true
    rm -rf "$path"
  fi
done

echo "Cleanup complete."
