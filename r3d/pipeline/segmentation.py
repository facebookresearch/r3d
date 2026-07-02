# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Segmentation data types for the R3D spatial AI pipeline.

ObjectSegmentation and FrameSegmentation represent the output of SAM3
segmentation. These types are the universal format used by all pipeline
stages -- online demo and offline evaluation share the same types.

Uses the pydantic v1 API for broad compatibility.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, validator


class ObjectSegmentation(BaseModel):
    """Segmentation result for one object in one frame."""

    object_id: int
    query_name: str
    bbox_2d: np.ndarray
    mask: np.ndarray
    mask_rle: dict[str, object]
    score: float
    obj_ptr: np.ndarray | None = None
    min_depth_m: float

    class Config:
        arbitrary_types_allowed = True

    @validator("bbox_2d")
    def validate_bbox(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError(f"bbox_2d must be np.ndarray, got {type(v).__name__}")
        if v.shape != (4,):
            raise ValueError(f"bbox_2d must be (4,), got {v.shape}")
        return v

    @validator("mask")
    def validate_mask(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError(f"mask must be np.ndarray, got {type(v).__name__}")
        if v.ndim != 2:
            raise ValueError(f"mask must be (H, W), got shape {v.shape}")
        return v

    @validator("obj_ptr")
    def validate_obj_ptr(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise TypeError(f"obj_ptr must be np.ndarray, got {type(v).__name__}")
            if v.ndim != 1:
                raise ValueError(f"obj_ptr must be 1-D, got shape {v.shape}")
        return v

    @validator("score")
    def validate_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {v}")
        return v

    @validator("mask_rle")
    def validate_rle(cls, v: dict[str, object]) -> dict[str, object]:
        if "counts" not in v or "size" not in v:
            raise ValueError("mask_rle must have 'counts' and 'size' keys")
        return v


class FrameSegmentation(BaseModel):
    """All segmentations for one frame (may span multiple queries)."""

    timestamp_ns: int
    objects: dict[int, ObjectSegmentation]

    class Config:
        arbitrary_types_allowed = True

    def filter_by_query(self, query_name: str) -> FrameSegmentation:
        """Return a new FrameSegmentation with only objects from the given query."""
        filtered = {
            oid: obj
            for oid, obj in self.objects.items()
            if obj.query_name == query_name
        }
        return FrameSegmentation(
            timestamp_ns=self.timestamp_ns,
            objects=filtered,
        )

    @property
    def query_names(self) -> set[str]:
        """All distinct query names present in this frame's segmentations."""
        return {obj.query_name for obj in self.objects.values()}

    @property
    def object_ids(self) -> list[int]:
        """All object IDs in this frame."""
        return list(self.objects.keys())
