#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/gin-gonic/gin /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 51aea73ba0f125f6cacc3b4b695efdf21d9c634f
git remote remove origin
go test -c .
