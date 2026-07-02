# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Sequence loading and processing utilities.

This module provides functions for loading and exploring ADT sequences.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
)

logger: logging.Logger = logging.getLogger(__name__)


def load_sequence_data(
    sequence_path: str,
    verbose: bool = True,
) -> Tuple[AriaDigitalTwinDataProvider, Any]:
    """Load ADT data for a given sequence.

    Args:
        sequence_path: Path to the sequence directory.
        verbose: If True, log loading information.

    Returns:
        Tuple of (gt_provider, data_paths).
    """
    if verbose:
        logger.info("Loading sequence: %s", Path(sequence_path).name)

    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    data_paths = paths_provider.get_datapaths()

    if verbose:
        logger.info("Data paths:")
        logger.info("  VRS file: %s", data_paths.aria_vrs_filepath)
        logger.info("  Depth images: %s", data_paths.depth_images_filepath)
        logger.info("  Segmentation: %s", data_paths.segmentations_filepath)
        logger.info("Loading ground truth data...")

    gt_provider = AriaDigitalTwinDataProvider(data_paths)

    if verbose:
        logger.info("Done loading ground truth data!")

    return gt_provider, data_paths


def get_valid_timestamps(
    gt_provider: AriaDigitalTwinDataProvider,
    stream_id: StreamId,
) -> list[int]:
    """Get timestamps that are within the ground truth data bounds.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        stream_id: Camera stream ID.

    Returns:
        List of valid timestamps in nanoseconds.
    """
    all_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
    gt_start_ns = gt_provider.get_start_time_ns()
    gt_end_ns = gt_provider.get_end_time_ns()

    valid_timestamps = [
        ts for ts in all_timestamps_ns if gt_start_ns <= ts <= gt_end_ns
    ]

    return valid_timestamps
