# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Local mesh path resolution for ADT object meshes.

Resolves object mesh paths from a local object library directory.
Expects meshes to be pre-downloaded into that directory.
"""

from __future__ import annotations

from pathlib import Path


def get_object_mesh_path(
    instance_name: str,
    object_library_path: str | None,
) -> str:
    """Get the local path to an object's GLB mesh.

    Looks for the mesh at ``{object_library_path}/{instance_name}/3d-asset.glb``.
    The ADT object library must be pre-downloaded locally.

    Args:
        instance_name: ADT instance name (e.g. "CoffeeCanisterSmall").
        object_library_path: Local directory containing the ADT object library.

    Returns:
        Path to the GLB mesh file.

    Raises:
        RuntimeError: If no object library is configured or the mesh is missing.
    """
    if object_library_path is None:
        raise RuntimeError(
            "No object library configured. Volume-based question types require "
            "the ADT object library; pass --object-library <dir> to generate_qa "
            "(the directory must contain {instance_name}/3d-asset.glb per object)."
        )

    local_path = Path(object_library_path) / instance_name / "3d-asset.glb"
    if not local_path.exists():
        raise RuntimeError(
            f"Mesh for '{instance_name}' not found at {local_path}. "
            f"Ensure the ADT object library at {object_library_path} is complete."
        )
    return str(local_path)
