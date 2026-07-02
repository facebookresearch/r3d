#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Full 8B eval: all annotations, no images, vLLM DP=8.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_eval_8b_full.sh <ASSETS_DIR> <OUTPUT_DIR>
set -euo pipefail

ASSETS=${1:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}
OUTPUT_DIR=${2:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}

mkdir -p "$OUTPUT_DIR"

echo "=== Generate Responses (8B, all annotations, DP=8) ==="
time python -u -m r3d.pipeline.scripts.generate_responses \
    --dataset facebook/r3d-bench \
    --scene-db "$ASSETS/scene/scene.db" \
    --frames-dir "$ASSETS/frames" \
    --model qwen3-vl-8b \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --backend vllm \
    --require-tracked-objects \
    --no-images \
    --data-parallel-size 8 \
    --concurrency 8 \
    --output-dir "$OUTPUT_DIR"

echo "=== Generate Scores ==="
python -u -m r3d.pipeline.scripts.generate_scores \
    --dataset facebook/r3d-bench \
    --responses-db "$OUTPUT_DIR/responses.db" \
    --output-dir "$OUTPUT_DIR"

echo "=== Results ==="
cat "$OUTPUT_DIR/eval_summary.txt" | head -50
echo "=== Done ==="
