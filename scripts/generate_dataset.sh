#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Regenerate the R3D-Bench Q&A annotations from ADT ground truth.
#
# This is the exact production command used to build R3D-Bench (settings from
# the paper run), adapted for OSS: local paths only, no Manifold/Hive.
#
# Most users do NOT need this — R3D-Bench is published on Hugging Face
# (facebook/r3d-bench) and downloaded automatically by the eval scripts. Use
# this only to reproduce or extend the benchmark.
#
# Requires:
#   $ADT_ROOT           ADT sequence directories (see README: Obtain ADT Sequences)
#   $ADT_OBJECT_LIBRARY ADT object meshes {name}/3d-asset.glb (see README: Obtain ADT Object Meshes)
#   $ASSETS/frames      extracted frames.db (see README: Extract Frames)
#
# Usage: ./generate_dataset.sh <OUTPUT_DIR>
set -euo pipefail

OUTPUT_DIR=${1:?Usage: $0 <OUTPUT_DIR>}
: "${ADT_ROOT:?set ADT_ROOT to your ADT sequences directory}"
: "${ADT_OBJECT_LIBRARY:?set ADT_OBJECT_LIBRARY to your ADT object-mesh directory}"
: "${ASSETS:?set ASSETS (must contain frames/frames.db)}"

SEQ_PATHS=$(find "$ADT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | tr '\n' ' ')

echo "=== Generate Dataset (production settings) ==="
time python -u -m r3d.data_gen.generate_qa \
    --sequences $SEQ_PATHS \
    --frames-db "$ASSETS/frames/frames.db" \
    --object-library "$ADT_OBJECT_LIBRARY" \
    --output-dir "$OUTPUT_DIR" \
    --questions-per-type 100 \
    --object-filter static \
    --disambiguation-method pointing \
    --require-unique-names \
    --min-fov-degrees 6.0 \
    --min-visibility-ratio 0.5 \
    --min-visible-frames 5

echo "=== Done ==="
