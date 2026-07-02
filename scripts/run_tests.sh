#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Run OSS unit tests (stdlib unittest -- no extra test dependency required).
# Assumes r3d is installed in the current environment (pip install -e .).
set -euo pipefail

python -m unittest discover -s tests/unit -p "test_*.py" -v
echo "=== Unit tests passed ==="
