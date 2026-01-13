#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/preactjs/preact /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard c12331064f4ea967641cd5e419204422af050fbb
git remote remove origin
npm install
