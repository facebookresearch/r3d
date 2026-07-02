# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Load R3D-Bench directly from the Hugging Face Hub as parquet-backed stores.

Downloads only the requested configs from the dataset repo (default
facebook/r3d-bench) and returns read-only parquet stores — no SQLite.
"""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download
from r3d.pipeline.stores.parquet_store import (
    ParquetAnnotationStore,
    ParquetMeshStore,
    ParquetSegmentationStore,
)

DEFAULT_REPO: str = "facebook/r3d-bench"


def download_configs(
    repo: str, configs: list[str], revision: str | None = None
) -> dict[str, str]:
    """Download the given configs' parquet files; return {config: local_path}."""
    patterns = [f"data/{c}/*.parquet" for c in configs]
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=patterns,
        revision=revision,
    )
    return {c: os.path.join(local, "data", c, f"{c}.parquet") for c in configs}


def load_annotation_store(
    repo: str = DEFAULT_REPO, revision: str | None = None
) -> ParquetAnnotationStore:
    paths = download_configs(repo, ["qa_annotations"], revision)
    return ParquetAnnotationStore(paths["qa_annotations"])


def load_segmentation_store(
    repo: str = DEFAULT_REPO, revision: str | None = None
) -> ParquetSegmentationStore:
    paths = download_configs(repo, ["segmentations"], revision)
    return ParquetSegmentationStore(paths["segmentations"])


def load_mesh_store(
    repo: str = DEFAULT_REPO, revision: str | None = None
) -> ParquetMeshStore:
    paths = download_configs(repo, ["meshes"], revision)
    return ParquetMeshStore(paths["meshes"])
