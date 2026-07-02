#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Build scene from frames and segmentations.
# Usage: ./run_build_scene.sh <ASSETS_DIR>
set -euo pipefail

ASSETS=${1:?Usage: $0 <ASSETS_DIR>}

echo "=== Build Scene ==="
time python -u -m r3d.pipeline.scripts.build_scene \
    --dataset facebook/r3d-bench \
    --frames-dir "$ASSETS/frames" \
    --output-dir "$ASSETS/scene" \
    --depth-stride 4 \
    --knn-k 6 \
    --mask-erosion 0 \
    --max-points-per-object 50000

echo "=== Done ==="
