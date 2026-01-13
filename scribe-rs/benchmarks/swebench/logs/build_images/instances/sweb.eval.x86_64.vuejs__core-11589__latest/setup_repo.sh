#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/vuejs/core /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 3653bc0f45d6fedf84e29b64ca52584359c383c0
git remote remove origin
pnpm i
