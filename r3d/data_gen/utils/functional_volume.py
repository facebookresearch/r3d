# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compute functional volume (capacity) of container objects.

Rule ("open cavity to the rim"): the capacity is the main cavity, i.e. the
largest connected empty region bounded by the container's inner wall, filled
up to the rim or to the lowest spill notch.

The mesh is rotated so its canonical up vector is +Z and voxelized at 2 mm
(surface voxelization; empty space is 6-connected, so a 26-connected surface
is watertight for it). Two kinds of candidate cavity are then found:

* Retained water: an empty voxel at height k holds water if, using only empty
  voxels at height <= k, it cannot reach the side or bottom of the grid. Water
  therefore fills each open cavity up to its rim or its lowest side opening
  (spout, slot, open side wall), never fills a recess that opens downwards
  (foot ring), and is not counted twice for an inverted object.
* Closed interior: empty space sealed off from the outside in 3D. Sealed
  components whose deepest voxel is at most ``MIN_INTERIOR_DEPTH_M`` from a
  surface are wall, base, handle, knob or lid material modelled as a closed
  shell (double surfaces) and never count. A thicker sealed component is the
  interior of a closed (lidded) container: the lid is treated as removed and
  the interior counts, up to the top of the container body. Head-space under
  a domed lid or a cap is excluded by cutting the interior above the highest
  horizontal slice whose area is at least ``LID_NARROWING_FRACTION`` of the
  largest slice area.

The capacity is the larger of the main retained-water component and the main
closed interior; everything else (other water pools, sealed pockets, hollow
knobs and handles) is not connected to the main cavity and is excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

MIN_INTERIOR_DEPTH_M: float = 0.012
LID_NARROWING_FRACTION: float = 0.5


@dataclass(frozen=True)
class CavityParts:
    """Voxel masks behind a functional-volume result (grid Z is up).

    Attributes:
        cavity: The counted main cavity.
        water: All retained water outside sealed regions (open cavities).
        sealed: All empty space sealed off from the outside in 3D.
        closed: True if the main cavity is a closed (lidded) interior.
    """

    cavity: np.ndarray
    water: np.ndarray
    sealed: np.ndarray
    closed: bool


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


def load_upright_mesh(
    glb_path: str, canonical_up: np.ndarray | None
) -> trimesh.Trimesh:
    """Load a GLB and rotate its canonical up vector (default +Y) to +Z."""
    mesh = trimesh.load(glb_path, force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise RuntimeError(f"GLB file has no faces: {glb_path}")
    up = np.array([0.0, 1.0, 0.0]) if canonical_up is None else canonical_up
    rot = _rotation_to_align(np.asarray(up, dtype=float), np.array([0.0, 0.0, 1.0]))
    mesh.vertices = (rot @ mesh.vertices.T).T
    return mesh


def compute_functional_volume(
    glb_path: str,
    canonical_up: np.ndarray | None = None,
    voxel_pitch_m: float = 0.002,
) -> float:
    """Compute the functional capacity of a container in cubic meters.

    Args:
        glb_path: Path to the 3d-asset.glb mesh file.
        canonical_up: The object's up vector in its local mesh frame
            (e.g. [0, 1, 0] from ADT instances.json canonical_pose).
            If None, defaults to [0, 1, 0].
        voxel_pitch_m: Voxel resolution in meters (default 2mm).

    Returns:
        Functional capacity in cubic meters (see the module docstring).

    Raises:
        RuntimeError: If the mesh cannot be loaded, has no faces, or has
            no cavity under the rule.
    """
    mesh = load_upright_mesh(glb_path, canonical_up)
    occupancy = np.asarray(mesh.voxelized(pitch=voxel_pitch_m).matrix, dtype=bool)
    volume = compute_cavity_volume_from_occupancy(
        occupancy, voxel_pitch_m, slice_axis=2
    )
    if volume == 0.0:
        raise RuntimeError(
            f"No cavity detected in mesh: {glb_path}. The object holds no "
            f"liquid under the open-cavity rule (no retained water and no "
            f"closed interior thicker than {2 * MIN_INTERIOR_DEPTH_M} m)."
        )
    return volume


def compute_cavity_volume_from_occupancy(
    occupancy: np.ndarray,
    voxel_pitch_m: float,
    slice_axis: int = 2,
) -> float:
    """Functional cavity volume (m^3) of a 3D surface-occupancy grid.

    Args:
        occupancy: 3D bool array where True = surface voxel.
        voxel_pitch_m: Size of each voxel in meters.
        slice_axis: Grid axis that points up (0=X, 1=Y, 2=Z).

    Returns:
        Cavity volume in cubic meters. Returns 0.0 if there is no cavity.
    """
    upright = np.moveaxis(np.asarray(occupancy, dtype=bool), slice_axis, 2)
    parts = functional_cavity_parts(upright, voxel_pitch_m)
    return cavity_voxel_count(parts.cavity, upright) * voxel_pitch_m**3


def cavity_voxel_count(cavity: np.ndarray, occupancy: np.ndarray) -> float:
    """Cavity voxels plus half of the surface voxels face-adjacent to them.

    A surface voxel is crossed by the wall at a uniformly distributed depth,
    so on average half of it lies on the cavity side. Without this term the
    2 mm grid under-reads capacity by about one voxel per wall.
    """
    if not cavity.any():
        return 0.0
    touching = ndimage.binary_dilation(cavity) & occupancy
    return float(cavity.sum()) + 0.5 * float(touching.sum())


def functional_cavity_parts(occupancy: np.ndarray, voxel_pitch_m: float) -> CavityParts:
    """Main-cavity mask and its candidate regions for a Z-up occupancy grid."""
    occ = np.pad(np.asarray(occupancy, dtype=bool), 1)
    sealed = _sealed_space(occ)
    water = _retained_water(occ) & ~sealed
    open_main = _largest_component(water, 0.0)
    min_depth_vox = MIN_INTERIOR_DEPTH_M / voxel_pitch_m
    closed_main = _cap_head_space(_largest_component(sealed, min_depth_vox))
    closed = int(closed_main.sum()) > int(open_main.sum())
    cavity = closed_main if closed else open_main
    inner = (slice(1, -1),) * 3
    return CavityParts(cavity[inner], water[inner], sealed[inner], closed)


def _sealed_space(occ: np.ndarray) -> np.ndarray:
    """Empty voxels that cannot reach the grid border in 3D."""
    labels, _ = ndimage.label(~occ)
    faces = [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
    faces += [labels[:, :, 0], labels[:, :, -1]]
    border = np.unique(np.concatenate([f.ravel() for f in faces]))
    return (labels > 0) & ~np.isin(labels, border)


def _retained_water(occ: np.ndarray) -> np.ndarray:
    """Empty voxels at height k that cannot reach the grid sides or bottom
    through empty voxels at height <= k (water filled to the spill level).

    Sweeps slices bottom-up, labelling each slice in 2D and merging the
    labels with the components of the slice below (union-find over component
    ids), which equals 6-connected labelling of every sub-volume [0, k].
    The grid must have an empty border (functional_cavity_parts pads it).
    """
    water = np.zeros_like(occ)
    forest = _ComponentForest()
    below = np.zeros(occ.shape[:2], dtype=np.int64)
    for k in range(occ.shape[2]):
        ids = forest.add_slice(~occ[:, :, k], bottom=k == 0)
        both = (ids > 0) & (below > 0)
        pairs = np.unique(np.stack([ids[both], below[both]], axis=1), axis=0)
        for a, b in pairs.tolist():
            forest.union(a, b)
        roots, outside = forest.resolve(ids)
        water[:, :, k] = (ids > 0) & ~outside
        below = roots
    return water


class _ComponentForest:
    """Union-find over per-slice 2D component ids with an "outside" flag."""

    def __init__(self) -> None:
        self.parent: list[int] = [0]
        self.outside: list[bool] = [False]

    def add_slice(self, empty: np.ndarray, bottom: bool) -> np.ndarray:
        labels, n = ndimage.label(empty)
        base = len(self.parent) - 1
        edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        touches = np.zeros(n + 1, dtype=bool)
        touches[edge] = True
        self.parent.extend(range(base + 1, base + n + 1))
        self.outside.extend(bool(bottom or touches[i]) for i in range(1, n + 1))
        return np.where(labels > 0, labels + base, 0)

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            self.outside[rb] = self.outside[rb] or self.outside[ra]

    def resolve(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Root id and outside flag per voxel of a slice's id map."""
        uniq, inverse = np.unique(ids, return_inverse=True)
        roots = np.array([self.find(int(i)) for i in uniq], dtype=np.int64)
        flags = np.array([self.outside[r] for r in roots], dtype=bool)
        shape = ids.shape
        return roots[inverse].reshape(shape), flags[inverse].reshape(shape)


def _largest_component(mask: np.ndarray, min_depth_vox: float) -> np.ndarray:
    """Largest 6-connected component whose deepest voxel (Euclidean distance
    to the nearest voxel outside the mask) exceeds min_depth_vox."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    index = np.arange(1, n + 1)
    sizes = np.asarray(ndimage.sum(mask, labels, index))
    if min_depth_vox > 0.0:
        depth = ndimage.distance_transform_edt(mask)
        max_depth = np.asarray(ndimage.maximum(depth, labels, index))
        sizes = np.where(max_depth > min_depth_vox, sizes, 0.0)
    if sizes.max() == 0.0:
        return np.zeros_like(mask)
    return labels == index[int(np.argmax(sizes))]


def _cap_head_space(interior: np.ndarray) -> np.ndarray:
    """Cut a closed interior above the highest slice whose area is at least
    LID_NARROWING_FRACTION of the largest slice area (domed lid, cap)."""
    if not interior.any():
        return interior
    area = interior.sum(axis=(0, 1))
    top = int(np.flatnonzero(area >= LID_NARROWING_FRACTION * area.max())[-1])
    capped = interior.copy()
    capped[:, :, top + 1 :] = False
    return capped
