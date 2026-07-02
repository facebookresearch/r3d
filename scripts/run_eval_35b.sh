#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 35B eval: all annotations, no images, vLLM DP=4.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_eval_35b.sh <ASSETS_DIR> <OUTPUT_DIR>
# Note: set FLASHINFER_CACHE_DIR to a local path (e.g. /tmp/flashinfer)
#       to avoid corrupted caches on shared filesystems.
set -euo pipefail

ASSETS=${1:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}
OUTPUT_DIR=${2:?Usage: $0 <ASSETS_DIR> <OUTPUT_DIR>}

export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_cache}
mkdir -p "$OUTPUT_DIR"

echo "=== Generate Responses (35B, all annotations, DP=4) ==="
time python -u -m r3d.pipeline.scripts.generate_responses \
    --dataset facebook/r3d-bench \
    --scene-db "$ASSETS/scene/scene.db" \
    --frames-dir "$ASSETS/frames" \
    --model qwen36-35b \
    --hf-model Qwen/Qwen3.6-35B-A3B \
    --backend vllm \
    --require-tracked-objects \
    --no-images \
    --data-parallel-size 4 \
    --concurrency 32 \
    --max-model-len 24576 \
    --output-dir "$OUTPUT_DIR"

echo "=== Generate Scores ==="
python -u -m r3d.pipeline.scripts.generate_scores \
    --dataset facebook/r3d-bench \
    --responses-db "$OUTPUT_DIR/responses.db" \
    --output-dir "$OUTPUT_DIR"

echo "=== Results ==="
head -50 "$OUTPUT_DIR/eval_summary.txt"
echo "=== Done ==="
