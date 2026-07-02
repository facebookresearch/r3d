# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Core frame data types for the R3D spatial AI pipeline.

FrameData is the universal per-frame representation used by all pipeline
stages. CameraIntrinsics replaces the untyped calibration objects used
throughout the codebase.

Uses the pydantic v1 API for broad compatibility.
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np
from pydantic import BaseModel, validator


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsics (post-undistortion).

    Extracted from projectaria_tools CameraCalibration via
    get_focal_lengths(), get_principal_point(), get_image_size().
    Downstream code never needs projectaria_tools.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    class Config:
        frozen = True

    @validator("width", "height")
    def positive_dimensions(cls, v: int) -> int:  # noqa: B902
        if v <= 0:
            raise ValueError(f"dimension must be positive, got {v}")
        return v

    @validator("fx", "fy")
    def positive_focal(cls, v: float) -> float:  # noqa: B902
        if v <= 0:
            raise ValueError(f"focal length must be positive, got {v}")
        return v


class DepthSource(enum.Enum):
    """Source of depth data in FrameData.

    NONE: No valid depth -- depth_map may contain a zero placeholder.
    GROUND_TRUTH: Depth from the data source (e.g., ADT).
    ESTIMATED: Depth from FoundationStereo or other estimation.
    """

    NONE = "NONE"
    GROUND_TRUTH = "GROUND_TRUTH"
    ESTIMATED = "ESTIMATED"


class FrameData(BaseModel):
    """Immutable per-frame input for the spatial AI pipeline.

    Universal across ADT sequences, Quest streams, and video files.
    """

    timestamp_ns: int
    rgb: np.ndarray
    depth_map: np.ndarray
    depth_source: DepthSource = DepthSource.GROUND_TRUTH
    intrinsics: CameraIntrinsics
    T_scene_device: np.ndarray
    T_device_camera: np.ndarray
    gravity_world: np.ndarray | None = None
    stereo_right: np.ndarray | None = None
    stereo_baseline_m: float | None = None

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    @validator("rgb")
    def validate_rgb(cls, v: np.ndarray) -> np.ndarray:  # noqa: B902
        if not isinstance(v, np.ndarray):
            raise TypeError(f"rgb must be np.ndarray, got {type(v).__name__}")
        if v.ndim != 3 or v.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {v.shape}")
        if v.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {v.dtype}")
        return v

    @validator("depth_map")
    def validate_depth(
        cls,  # noqa: B902
        v: np.ndarray,
        values: dict[str, Any],
    ) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError(f"depth_map must be np.ndarray, got {type(v).__name__}")
        if v.ndim != 2:
            raise ValueError(f"depth_map must be (H, W), got shape {v.shape}")
        if v.dtype not in (np.float32, np.float64):
            raise ValueError(f"depth_map must be float32/float64, got {v.dtype}")
        if v.size == 0:
            raise ValueError(f"depth_map must be non-empty, got shape {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("depth_map must be finite (no NaN/inf)")
        if np.any(v < 0):
            raise ValueError(
                f"depth_map must be non-negative, got min {float(v.min())}"
            )
        if "rgb" in values and v.shape != values["rgb"].shape[:2]:
            raise ValueError(
                f"depth_map (H, W)={v.shape} must match "
                f"rgb (H, W)={values['rgb'].shape[:2]}"
            )
        return v

    @validator("T_scene_device", "T_device_camera")
    def validate_se3(cls, v: np.ndarray) -> np.ndarray:  # noqa: B902
        if not isinstance(v, np.ndarray):
            raise TypeError(f"SE3 transform must be np.ndarray, got {type(v).__name__}")
        if v.shape != (4, 4):
            raise ValueError(f"SE3 transform must be (4, 4), got {v.shape}")
        return v

    @validator("gravity_world")
    def validate_gravity(cls, v: np.ndarray | None) -> np.ndarray | None:  # noqa: B902
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"gravity_world must be np.ndarray, got {type(v).__name__}"
                )
            if v.shape != (3,):
                raise ValueError(f"gravity_world must be (3,), got {v.shape}")
        return v

    @validator("stereo_right")
    def validate_stereo_right(
        cls,  # noqa: B902
        v: np.ndarray | None,
        values: dict[str, Any],
    ) -> np.ndarray | None:
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"stereo_right must be np.ndarray, got {type(v).__name__}"
                )
            if v.ndim != 3 or v.shape[2] != 3:
                raise ValueError(f"stereo_right must be (H, W, 3), got {v.shape}")
            if v.dtype != np.uint8:
                raise ValueError(f"stereo_right must be uint8, got {v.dtype}")
            if "rgb" in values and v.shape[:2] != values["rgb"].shape[:2]:
                raise ValueError(
                    f"stereo_right (H, W)={v.shape[:2]} must match "
                    f"rgb (H, W)={values['rgb'].shape[:2]}"
                )
        return v

    @validator("stereo_baseline_m")
    def validate_stereo_baseline(cls, v: float | None) -> float | None:  # noqa: B902
        if v is not None and v <= 0:
            raise ValueError(f"stereo_baseline_m must be positive, got {v}")
        return v
