# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Rigid transform inversion utilities (numpy only).

Provides exact inverse of 4x4 rigid transforms using the R^T / -R^T t
identity rather than generic matrix inversion, avoiding numerical error
accumulation.
"""

from __future__ import annotations

import math

import numpy as np


def invert_rigid_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform without np.linalg.inv().

    For T = [R | t; 0 1], the inverse is [R^T | -R^T @ t; 0 1].
    R^T is exact (transpose), so no numerical error accumulates.
    """
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -(T[:3, :3].T @ T[:3, 3])
    return T_inv


def rotation_matrix_to_quaternion(R: np.ndarray | list[list[float]]) -> list[float]:
    """Convert a 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Uses Shepperd's method for numerical stability. Accepts either a
    numpy array or a nested list of floats.
    """
    if not isinstance(R, np.ndarray):
        R = np.array(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return [float(qw), float(qx), float(qy), float(qz)]
