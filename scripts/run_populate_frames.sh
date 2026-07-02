#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Populate frames from ADT sequences.
# Usage: ./run_populate_frames.sh <ADT_ROOT> <OUTPUT_DIR> [NUM_WORKERS]
set -euo pipefail

ADT_ROOT=${1:?Usage: $0 <ADT_ROOT> <OUTPUT_DIR> [NUM_WORKERS]}
OUTPUT_DIR=${2:?Usage: $0 <ADT_ROOT> <OUTPUT_DIR> [NUM_WORKERS]}
NUM_WORKERS=${3:-16}

mapfile -t SEQ_PATHS < <(find "$ADT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)

echo "=== Populate Frames ==="
echo "ADT root: $ADT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS"

time python -u -m r3d.pipeline.scripts.populate_frames \
    --sequence-paths "${SEQ_PATHS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --fps 3.0 \
    --image-size 512 \
    --num-workers "$NUM_WORKERS"

echo "=== Done ==="
