#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/astral-sh/ruff /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 75a24bbc67aa31b825b6326cfb6e6afdf3ca90d5
git remote remove origin
RUSTFLAGS=-Awarnings cargo test --package ruff_linter --no-run
