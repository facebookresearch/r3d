#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Run OSS unit and integration tests (stdlib unittest -- no extra test dependency required).
# Assumes r3d is installed in the current environment (pip install -e .).
set -euo pipefail

python -m unittest discover -s tests -t . -p "test_*.py" -v
echo "=== Tests passed ==="
