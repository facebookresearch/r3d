#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# RGB baseline eval: VLM sees multi-frame images (+ SAM3 mask overlays), no 3D
# scene or tools. This is the production RGB-baseline command.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_rgb_baseline.sh <ASSETS_DIR> <OUTPUT_DIR>
set -euo pipefail

ASSETS=${1:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}
OUTPUT_DIR=${2:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}

mkdir -p "$OUTPUT_DIR"

echo "=== RGB baseline (Qwen3-VL-8B, overlay, DP=8) ==="
time python -u -m r3d.scripts.rgb_overlay_evals \
    --dataset facebook/r3d-bench \
    --frames-dir "$ASSETS/frames" \
    --model qwen3-vl-8b \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --backend vllm \
    --data-parallel-size 8 \
    --overlay \
    --score \
    --output-dir "$OUTPUT_DIR"

echo "=== Done ==="
