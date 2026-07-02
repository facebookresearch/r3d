# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared ADT (Aria Digital Twin) provider loading.

Centralises the 3-line pattern of instantiating
AriaDigitalTwinDataPathsProvider -> get_datapaths -> AriaDigitalTwinDataProvider
that was previously copy-pasted across many call-sites.
"""

from __future__ import annotations

import logging
from typing import Any

from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
)

logger: logging.Logger = logging.getLogger(__name__)


def load_adt_provider(
    sequence_path: str,
    verbose: bool = False,
) -> tuple[AriaDigitalTwinDataProvider, Any]:
    """Load an ADT sequence and return its data provider.

    Args:
        sequence_path: Path to the ADT sequence directory.
        verbose: If True, log the loaded path at INFO level.

    Returns:
        Tuple of (data_provider, data_paths).

    Raises:
        RuntimeError: If the sequence cannot be loaded.
    """
    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
        data_paths = paths_provider.get_datapaths()
        provider = AriaDigitalTwinDataProvider(data_paths)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ADT sequence at {sequence_path}: {e}"
        ) from e

    if verbose:
        logger.info(f"Loaded ADT sequence: {sequence_path}")

    return provider, data_paths
