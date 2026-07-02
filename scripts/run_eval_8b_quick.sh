#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Quick 8B eval: 100 annotations, no images, vLLM DP=1.
# Usage: ./run_eval_8b_quick.sh <ASSETS_DIR> <OUTPUT_DIR>
set -euo pipefail

ASSETS=${1:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}
OUTPUT_DIR=${2:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}

mkdir -p "$OUTPUT_DIR"

echo "=== Generate Responses (8B, 100 annotations) ==="
time python -u -m r3d.pipeline.scripts.generate_responses \
    --dataset facebook/r3d-bench \
    --scene-db "$ASSETS/scene/scene.db" \
    --frames-dir "$ASSETS/frames" \
    --model qwen3-vl-8b \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --backend vllm \
    --require-tracked-objects \
    --no-images \
    --data-parallel-size 1 \
    --concurrency 1 \
    --max-annotations 100 \
    --output-dir "$OUTPUT_DIR"

echo "=== Generate Scores ==="
python -u -m r3d.pipeline.scripts.generate_scores \
    --dataset facebook/r3d-bench \
    --responses-db "$OUTPUT_DIR/responses.db" \
    --output-dir "$OUTPUT_DIR"

echo "=== Results ==="
head -40 "$OUTPUT_DIR/eval_summary.txt"
echo "=== Done ==="
