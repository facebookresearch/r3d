# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compute functional volume (capacity) of container objects.

Uses voxel-based cavity detection via 2D slice integration: for each
horizontal slice through the voxelized mesh, detect enclosed interior
regions using binary_fill_holes, then integrate across slices. This
gives the volume of liquid a container can hold.

The mesh is oriented upright using its canonical up_vector (from the
ADT instances.json) so that slicing along Z correctly captures the
holdable volume regardless of the object's scene placement.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage


def _rotation_to_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix that rotates unit vector src to dst."""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        perp = np.array([1, 0, 0]) if abs(src[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, src) * src
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + vx + vx @ vx / (1 + c)


def compute_functional_volume(
    glb_path: str,
    canonical_up: np.ndarray | None = None,
    voxel_pitch_m: float = 0.002,
) -> float:
    """Compute the functional capacity of a container in cubic meters.

    The mesh is rotated so its canonical up vector aligns with +Z
    (upright orientation), then sliced horizontally.

    Args:
        glb_path: Path to the 3d-asset.glb mesh file.
        canonical_up: The object's up vector in its local mesh frame
            (e.g. [0, 1, 0] from ADT instances.json canonical_pose).
            If None, defaults to [0, 1, 0].
        voxel_pitch_m: Voxel resolution in meters (default 2mm).

    Returns:
        Functional capacity in cubic meters.

    Raises:
        RuntimeError: If the mesh cannot be loaded, has no faces, or has
            no detectable interior cavity.
    """
    mesh = trimesh.load(glb_path, force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RuntimeError(f"GLB file has no faces: {glb_path}")

    if canonical_up is None:
        canonical_up = np.array([0.0, 1.0, 0.0])
    else:
        canonical_up = np.asarray(canonical_up, dtype=float)

    R = _rotation_to_align(canonical_up, np.array([0.0, 0.0, 1.0]))
    mesh.vertices = (R @ mesh.vertices.T).T

    voxelized = mesh.voxelized(pitch=voxel_pitch_m)
    occupancy = voxelized.matrix

    volume = compute_cavity_volume_from_occupancy(
        occupancy, voxel_pitch_m, slice_axis=2
    )
    if volume == 0.0:
        raise RuntimeError(
            f"No interior cavity detected in mesh: {glb_path}. "
            f"The object may not be a container or the mesh walls "
            f"may be too thin for the voxel resolution ({voxel_pitch_m}m)."
        )
    return volume


def compute_cavity_volume_from_occupancy(
    occupancy: np.ndarray,
    voxel_pitch_m: float,
    slice_axis: int = 2,
) -> float:
    """Compute interior cavity volume from a 3D binary occupancy grid.

    For each slice along slice_axis, uses binary_fill_holes to detect
    enclosed interior regions, then integrates across slices.

    Args:
        occupancy: 3D bool array where True = solid.
        voxel_pitch_m: Size of each voxel in meters.
        slice_axis: Axis to slice along (0=X, 1=Y, 2=Z).

    Returns:
        Cavity volume in cubic meters. Returns 0.0 if no cavity detected.
    """
    cavity_voxel_count = 0
    for i in range(occupancy.shape[slice_axis]):
        slc = [slice(None)] * 3
        slc[slice_axis] = i
        plane = occupancy[tuple(slc)]
        if not plane.any():
            continue
        filled_2d = ndimage.binary_fill_holes(plane)
        interior = filled_2d & ~plane
        cavity_voxel_count += int(interior.sum())

    return cavity_voxel_count * voxel_pitch_m**3
