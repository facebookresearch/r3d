# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared logging configuration for the R3D pipeline.

Call ``setup_logging()`` once at the entry point of each script.
All modules should use ``logging.getLogger(__name__)`` as usual --
this module only configures the root logger's format and level.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
) -> None:
    """Configure root logger with timestamps and consistent format.

    Args:
        level: Logging level.
        log_dir: If set, also write logs to ``{log_dir}/run.log``.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "run.log")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(fh)
