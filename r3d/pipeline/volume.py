# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Object volume estimation from the SAM3D mesh rescaled to the fitted OBB.

Volume is the mesh cavity (functional) volume after rescaling the SAM3D mesh to
match the OBB dimensions (``mesh_rescaled``). ``bbox_volume`` is an internal
helper (OBB AABB volume) used to compute the rescale ratio.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh
from r3d.data_gen.utils.functional_volume import compute_functional_volume


def bbox_volume(obb_aabb: np.ndarray) -> float:
    """Volume from OBB AABB [xmin, xmax, ymin, ymax, zmin, zmax]."""
    return float(
        (obb_aabb[1] - obb_aabb[0])
        * (obb_aabb[3] - obb_aabb[2])
        * (obb_aabb[5] - obb_aabb[4])
    )


def rescale_mesh_to_obb(mesh_path: str, obb_aabb: np.ndarray):
    """Load a GLB mesh and rescale it to match the OBB's bbox volume.

    Computes the ratio of OBB bbox volume to mesh bbox volume, derives a linear
    scale factor (cube root), and rescales the mesh vertices in place.

    Args:
        mesh_path: Path to the .glb mesh file.
        obb_aabb: (6,) array [xmin, xmax, ymin, ymax, zmin, zmax].

    Returns:
        The rescaled trimesh.Trimesh, or None if the mesh has no faces or a
        degenerate bounding box.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        return None

    obb_vol = bbox_volume(obb_aabb)
    mesh_extents = mesh.bounding_box.extents
    mesh_bbox_vol = float(mesh_extents[0] * mesh_extents[1] * mesh_extents[2])
    if mesh_bbox_vol <= 0:
        return None

    scale = (obb_vol / mesh_bbox_vol) ** (1.0 / 3.0)
    mesh.vertices *= scale
    return mesh


def mesh_rescaled_volume(mesh_path: str, obb_aabb: np.ndarray) -> float | None:
    """SAM3D mesh cavity volume rescaled to match OBB dimensions.

    Rescales the mesh to the OBB (see rescale_mesh_to_obb), then computes
    functional (cavity) volume on the rescaled mesh.

    Args:
        mesh_path: Path to the .glb mesh file.
        obb_aabb: (6,) array [xmin, xmax, ymin, ymax, zmin, zmax].

    Returns:
        Rescaled cavity volume in cubic meters, or None if no cavity.
    """
    mesh = rescale_mesh_to_obb(mesh_path, obb_aabb)
    if mesh is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        mesh.export(tmp_path)
        return compute_functional_volume(tmp_path)
    except RuntimeError:
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
