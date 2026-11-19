# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Build scene.db from frames.db and the R3D-Bench segmentations.

Replaces the internal orchestrator + gsplat + scene_builder chain with a
simpler depth-lifting pipeline:

1. Load frames from frames.db and segmentations from the HF dataset
2. For each segmented object across all frames:
   - Depth-lift masked pixels to 3D world coordinates
3. KNN outlier filtering and multi-view consensus voting on the aggregated
   3D points (each step can be disabled for ablations)
4. Fit gravity-aligned oriented bounding boxes (OBB)
5. Write per-object 3D points, scene objects, reconstructions, frame
   visibility, and coverage to scene.db

Ablation modes (defaults reproduce the paper scene exactly):
    --no-knn             skip both KNN outlier-removal passes
    --no-multiview-vote  skip the multi-view consensus vote
    --box-mode median    no multi-view fusion: fit one OBB per frame from that
                         frame's points only, then take the per-object median
                         box (see _median_box)
    --box-mode first_frame
                         no multi-view fusion: fit one OBB to the points of the
                         object's earliest mask frame with enough points after
                         per-frame KNN (see _first_frame_box)

Usage:
    python -m r3d.pipeline.scripts.build_scene \
      --dataset facebook/r3d-bench \
      --frames-dir /tmp/eval \
      --output-dir /tmp/eval
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from r3d.pipeline.eval.config import DEFAULT_DATASET
from r3d.pipeline.frame_data import CameraIntrinsics
from r3d.pipeline.hf_dataset import load_segmentation_store
from r3d.pipeline.scene_builder import build_scene
from r3d.pipeline.stores.base import (
    ObjectReconstruction,
    SceneObject,
    SegmentationStore,
)
from r3d.pipeline.stores.sqlite_store import (
    SQLiteFrameStore,
    SQLiteObjectPointsStore,
    SQLiteSceneStore,
)
from r3d.utils.logging import setup_logging
from scipy.spatial import cKDTree

logger: logging.Logger = logging.getLogger(__name__)


def _lift_depth_to_3d(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_scene_device: np.ndarray,
    T_device_camera: np.ndarray,
    stride: int,
) -> np.ndarray:
    h, w = depth.shape[:2]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    vs = vs.ravel()
    us = us.ravel()

    mask_sub = mask[vs, us].astype(bool)
    depth_sub = depth[vs, us]

    valid = mask_sub & (depth_sub > 0)
    us_valid = us[valid]
    vs_valid = vs[valid]
    d_valid = depth_sub[valid]

    if len(d_valid) == 0:
        return np.empty((0, 3), dtype=np.float64)

    x_cam = (us_valid - intrinsics.cx) / intrinsics.fx * d_valid
    y_cam = (vs_valid - intrinsics.cy) / intrinsics.fy * d_valid
    z_cam = d_valid

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
    T_scene_camera = T_scene_device @ T_device_camera
    pts_world = (T_scene_camera @ pts_cam.T).T[:, :3]

    return pts_world


def _filter_outliers_knn(
    points: np.ndarray, k: int = 6, threshold_multiplier: float = 3.0
) -> np.ndarray:
    if len(points) <= max(k, 20):
        return points

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(k, len(points)))
    median_nn_dist = np.median(dists[:, 1])
    mean_knn = dists[:, 1:].mean(axis=1)

    inlier_mask = mean_knn < threshold_multiplier * median_nn_dist
    return points[inlier_mask]


def _multiview_consensus_filter(
    points: np.ndarray,
    obj_id: int,
    sequence_id: str,
    frame_store: "SQLiteFrameStore",
    seg_store: "SegmentationStore",
) -> np.ndarray:
    """Filter 3D points by multi-view reprojection consensus.

    Replicates the production gsplat identity-voting membership test. Each
    point is reprojected into every view; a point is kept iff it lands inside
    the object's SAM3 mask in strictly more than half of the views where it is
    in-bounds. In the production run the per-point identity label map contains
    only two labels (this object vs. background), so the classifier argmax
    reduces exactly to this >50% mask-membership vote. No depth/occlusion test
    is applied (disabled in the production config).
    """
    N = len(points)
    if N == 0:
        return points

    timestamps = frame_store.get_all_timestamps(sequence_id)
    # Only the query names that actually contain this object — avoids scanning
    # (and RLE-decoding) every query's segmentation on every frame.
    query_names = seg_store.get_query_names_for_object(sequence_id, obj_id)
    if not query_names:
        return points

    obj_votes = np.zeros(N, dtype=np.int32)
    total_visible = np.zeros(N, dtype=np.int32)

    ones = np.ones((N, 1))
    pts_h = np.hstack([points, ones])

    for ts in timestamps:
        # Only vote over frames where this object is actually segmented — this
        # matches production, whose per-object cache contains exactly those
        # frames. Counting frames where the object is absent would make every
        # point vote "background" and wrongly fail the majority test.
        obj_mask_union: np.ndarray | None = None
        for qn in query_names:
            seg = seg_store.get_segmentation(sequence_id, ts, qn)
            if obj_id in seg.objects:
                mask = seg.objects[obj_id].mask
                if obj_mask_union is None:
                    obj_mask_union = np.zeros(mask.shape, dtype=bool)
                if mask.shape == obj_mask_union.shape:
                    obj_mask_union |= mask
        if obj_mask_union is None:
            continue

        intrinsics, t_scene_device, t_device_camera = frame_store.load_frame_pose(
            sequence_id, ts
        )
        T_scene_camera = t_scene_device @ t_device_camera
        T_camera_scene = np.linalg.inv(T_scene_camera)

        pts_cam = (T_camera_scene @ pts_h.T).T[:, :3]
        depth = pts_cam[:, 2]
        valid = depth > 0

        if not valid.any():
            continue

        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy

        px = np.zeros(N)
        py = np.zeros(N)
        px[valid] = fx * pts_cam[valid, 0] / depth[valid] + cx
        py[valid] = fy * pts_cam[valid, 1] / depth[valid] + cy

        w = intrinsics.width
        h = intrinsics.height
        ix = np.round(px).astype(np.int32)
        iy = np.round(py).astype(np.int32)
        in_bounds = valid & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        if not in_bounds.any():
            continue

        total_visible[in_bounds] += 1

        ib_idx = np.where(in_bounds)[0]
        in_mask = obj_mask_union[iy[in_bounds], ix[in_bounds]]
        obj_votes[ib_idx[in_mask]] += 1

    visible = total_visible > 0
    keep = visible & (obj_votes > (total_visible / 2.0))
    return points[keep]


def _fit_gravity_aligned_obb(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a gravity-aligned OBB matching the production gsplat_trainer.

    Replicates simple_trainer._fit_gravity_aligned_obb exactly:
    PCA on XZ plane, project onto principal axes, symmetric AABB,
    position = mean of corners.
    """
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    xz = points[:, [0, 2]]
    xz_mean = xz.mean(axis=0)
    xz_centered = xz - xz_mean
    cov = np.cov(xz_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    axes_2d = eigvecs[:, order]

    proj = xz_centered @ axes_2d
    p_min = proj.min(axis=0)
    p_max = proj.max(axis=0)

    extent = np.array([p_max[0] - p_min[0], y_max - y_min, p_max[1] - p_min[1]])

    face_locals = [
        np.array([p_min[0], p_min[1]]),
        np.array([p_max[0], p_min[1]]),
        np.array([p_max[0], p_max[1]]),
        np.array([p_min[0], p_max[1]]),
    ]
    corners = []
    for y in [y_min, y_max]:
        for local in face_locals:
            world_xz = local @ axes_2d.T + xz_mean
            corners.append([world_xz[0], y, world_xz[1]])
    corners = np.array(corners)

    position = corners.mean(axis=0).astype(np.float64)

    e0 = corners[1] - corners[0]
    e1 = corners[4] - corners[0]
    e2 = corners[3] - corners[0]
    e0 /= np.linalg.norm(e0) + 1e-8
    e1 /= np.linalg.norm(e1) + 1e-8
    e2 /= np.linalg.norm(e2) + 1e-8

    obb_transform = np.eye(4, dtype=np.float64)
    obb_transform[:3, 0] = e0
    obb_transform[:3, 1] = e1
    obb_transform[:3, 2] = e2
    obb_transform[:3, 3] = position

    obb_aabb = np.array(
        [
            -extent[0] / 2,
            extent[0] / 2,
            -extent[1] / 2,
            extent[1] / 2,
            -extent[2] / 2,
            extent[2] / 2,
        ],
        dtype=np.float64,
    )

    return obb_transform, obb_aabb, position


def _canonical_box_params(
    obb_transform: np.ndarray, obb_aabb: np.ndarray
) -> tuple[np.ndarray, float, float, float, float]:
    """Reduce a gravity-aligned OBB to (center, long, short, height, yaw).

    The PCA in _fit_gravity_aligned_obb leaves two ambiguities: the sign of
    each horizontal axis (yaw ambiguous by pi) and, for near-square
    footprints, which axis is the major one. Sorting the horizontal extents
    removes the order ambiguity; yaw is the angle of the long axis in the XZ
    plane reduced modulo pi.
    """
    extent_0 = float(obb_aabb[1] - obb_aabb[0])
    height = float(obb_aabb[3] - obb_aabb[2])
    extent_2 = float(obb_aabb[5] - obb_aabb[4])
    long_axis = obb_transform[:3, 0] if extent_0 >= extent_2 else obb_transform[:3, 2]
    yaw = float(np.mod(np.arctan2(long_axis[2], long_axis[0]), np.pi))
    center = obb_transform[:3, 3].astype(np.float64)
    return center, max(extent_0, extent_2), min(extent_0, extent_2), height, yaw


def _median_yaw(yaws: np.ndarray) -> float:
    """Median of axis angles defined modulo pi.

    Doubling maps the pi-periodic yaw onto the full circle; the circular mean
    of the doubled angles is the reference, each yaw is unwrapped to within
    +/- pi/2 of that reference, and the ordinary median of the unwrapped
    yaws is returned (reduced back to [0, pi)).
    """
    doubled = 2.0 * yaws
    ref = 0.5 * float(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean()))
    deviation = np.mod(yaws - ref + np.pi / 2.0, np.pi) - np.pi / 2.0
    return float(np.mod(ref + np.median(deviation), np.pi))


def _median_box(
    boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elementwise median of per-frame gravity-aligned boxes.

    Takes the median of the centers (per coordinate), of the sorted
    horizontal extents (long, short), of the height, and of the long-axis
    yaw (see _median_yaw). Returns (obb_transform, obb_aabb, position) in
    the same layout as _fit_gravity_aligned_obb: column 0 is the long axis,
    column 1 is gravity (+Y), column 2 completes a right-handed frame.
    """
    if not boxes:
        raise ValueError("_median_box requires at least one box")
    params = [_canonical_box_params(t, a) for t, a, _ in boxes]
    center = np.median(np.stack([p[0] for p in params]), axis=0)
    long_ext = float(np.median([p[1] for p in params]))
    short_ext = float(np.median([p[2] for p in params]))
    height = float(np.median([p[3] for p in params]))
    yaw = _median_yaw(np.array([p[4] for p in params]))

    obb_transform = np.eye(4, dtype=np.float64)
    obb_transform[:3, 0] = [np.cos(yaw), 0.0, np.sin(yaw)]
    obb_transform[:3, 1] = [0.0, 1.0, 0.0]
    obb_transform[:3, 2] = [-np.sin(yaw), 0.0, np.cos(yaw)]
    obb_transform[:3, 3] = center
    half = np.array([long_ext, height, short_ext], dtype=np.float64) / 2.0
    obb_aabb = np.array(
        [-half[0], half[0], -half[1], half[1], -half[2], half[2]], dtype=np.float64
    )
    return obb_transform, obb_aabb, center.copy()


def _median_box_from_frames(
    frame_chunks: list[np.ndarray],
    knn_k: int,
    use_knn: bool,
    min_frame_points: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, list[np.ndarray]]:
    """Fit one OBB per frame (no cross-frame fusion) and return the median box.

    Each chunk holds the points lifted from a single frame's mask. Frames
    with fewer than ``min_frame_points`` points after (optional) KNN
    filtering are dropped. Returns (median_box or None if no frame is
    valid, filtered points of each valid frame).
    """
    boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    kept: list[np.ndarray] = []
    for chunk in frame_chunks:
        pts = _filter_outliers_knn(chunk, knn_k) if use_knn else chunk
        if len(pts) < min_frame_points:
            continue
        box = _fit_gravity_aligned_obb(pts)
        if box is None:
            raise RuntimeError("_fit_gravity_aligned_obb returned None")
        boxes.append(box)
        kept.append(pts)
    if not boxes:
        return None, kept
    return _median_box(boxes), kept


def _first_frame_box(
    frame_chunks: list[tuple[int, np.ndarray]],
    knn_k: int,
    use_knn: bool,
    min_frame_points: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, int, int]:
    """Fit one OBB to the object's first valid detected frame (no fusion).

    ``frame_chunks`` holds (timestamp_ns, lifted points) for every frame with
    a stored mask, including frames whose mask lifts to no valid depth. Frames
    are visited in timestamp order; the first one with >= ``min_frame_points``
    points after the per-frame KNN of --box-mode median (the same validity
    rule as that mode) is fitted alone. Raises if no frame qualifies or two
    masks share a timestamp. Returns (box, that frame's filtered points, its
    timestamp, its index among the object's mask frames; 0 = first mask).
    """
    stamps = [t for t, _ in frame_chunks]
    if len(set(stamps)) != len(stamps):
        raise RuntimeError("two masks share a timestamp")
    for idx, (ts, chunk) in enumerate(sorted(frame_chunks, key=lambda c: c[0])):
        pts = _filter_outliers_knn(chunk, knn_k) if use_knn and len(chunk) else chunk
        if len(pts) < min_frame_points:
            continue
        box = _fit_gravity_aligned_obb(pts)
        if box is None:
            raise RuntimeError("_fit_gravity_aligned_obb returned None")
        return box, pts, ts, idx
    raise RuntimeError(
        f"none of {len(frame_chunks)} mask frames has >= {min_frame_points} "
        "points after KNN"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build scene.db from frames.db + segmentations.db via depth lifting"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing frames.db (and frame_data/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="HF dataset repo for R3D-Bench SAM3 segmentations (parquet). "
        f"Default: {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for scene.db and object_points.db.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=6,
        help="Number of nearest neighbors for outlier filtering (default: 6).",
    )
    parser.add_argument(
        "--max-points-per-object",
        type=int,
        default=50000,
        help="Maximum 3D points per object after aggregation (default: 50000). Randomly subsampled if exceeded.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=6,
        help="Minimum 3D points required to fit an OBB (default: 6).",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=4,
        help="Subsample stride for depth lifting (default: 4, 1 = every pixel).",
    )
    parser.add_argument(
        "--mask-erosion",
        type=int,
        default=0,
        help="Erode SAM3 masks by this many pixels before depth lifting (default: 0).",
    )
    parser.add_argument(
        "--no-knn",
        action="store_true",
        help="Ablation: skip both KNN outlier-removal passes (and the per-frame "
        "KNN pass in --box-mode median).",
    )
    parser.add_argument(
        "--no-multiview-vote",
        action="store_true",
        help="Ablation: skip the multi-view consensus vote. Required with "
        "--box-mode median, which has no cross-frame fusion to vote over.",
    )
    parser.add_argument(
        "--box-mode",
        choices=["fused", "median", "first_frame"],
        default="fused",
        help="fused (default): fit one OBB to the points fused from all frames. "
        "median: fit one OBB per frame and take the per-object median box. "
        "first_frame: fit one OBB to the object's earliest mask frame with "
        "enough points after per-frame KNN.",
    )
    parser.add_argument(
        "--median-min-frame-points",
        type=int,
        default=10,
        help="--box-mode median/first_frame only: minimum points (after "
        "per-frame KNN) a frame needs (default: 10; at "
        "--depth-stride 4 many small objects never reach 50 points in any "
        "single frame).",
    )
    parser.add_argument(
        "--sequence-ids",
        nargs="+",
        default=None,
        help="Only build these sequences (default: every sequence in frames.db).",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.box_mode != "fused" and not args.no_multiview_vote:
        parser.error(
            f"--box-mode {args.box_mode} has no cross-frame fusion, so the "
            "multi-view vote cannot run; pass --no-multiview-vote explicitly."
        )
    if args.median_min_frame_points < 3:
        parser.error("--median-min-frame-points must be >= 3 to fit an OBB")


def _erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
    )
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _collect_object_points(
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    seg_store: SegmentationStore,
    depth_stride: int,
    mask_erosion: int = 8,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """obj_id -> [(timestamp_ns, lifted points)] for every frame with a mask.

    Frames whose mask lifts to no valid depth are kept with an empty array.
    """
    timestamps = frame_store.get_all_timestamps(sequence_id)
    query_names = seg_store.get_all_query_names(sequence_id)
    object_points: dict[int, list[tuple[int, np.ndarray]]] = {}
    seen: set[tuple[int, int]] = set()

    for ts in timestamps:
        frame = frame_store.load_frame(sequence_id, ts)
        for query_name in query_names:
            seg = seg_store.get_segmentation(sequence_id, ts, query_name)
            for obj_id, obj_seg in seg.objects.items():
                if (ts, obj_id) in seen:
                    continue
                seen.add((ts, obj_id))
                mask = obj_seg.mask
                if mask_erosion > 0:
                    mask = _erode_mask(mask, mask_erosion)
                pts = _lift_depth_to_3d(
                    frame.depth_map,
                    mask,
                    frame.intrinsics,
                    frame.T_scene_device,
                    frame.T_device_camera,
                    depth_stride,
                )
                object_points.setdefault(obj_id, []).append((ts, pts))

    return object_points


@dataclass(frozen=True)
class ObjectBoxConfig:
    """Per-object point filtering and box fitting recipe.

    The defaults reproduce the paper scene: fused points, init-KNN ->
    multi-view vote -> bbox-KNN -> one gravity-aligned OBB.
    """

    knn_k: int = 6
    use_knn: bool = True
    use_vote: bool = True
    box_mode: str = "fused"
    median_min_frame_points: int = 10
    max_points_per_object: int = 50000


def _subsample(points: np.ndarray, obj_id: int, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.RandomState(obj_id)
    indices = rng.choice(len(points), max_points, replace=False)
    return points[indices]


def _fused_object_points(
    all_pts: np.ndarray,
    obj_id: int,
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    seg_store: SegmentationStore,
    cfg: ObjectBoxConfig,
) -> np.ndarray:
    # Production recipe: init-KNN (prune_init_knn) -> multiview membership
    # vote (identity argmax) -> bbox-KNN (bbox_method="knn") -> OBB fit.
    init_filtered = _filter_outliers_knn(all_pts, cfg.knn_k) if cfg.use_knn else all_pts
    consensus = (
        _multiview_consensus_filter(
            init_filtered, obj_id, sequence_id, frame_store, seg_store
        )
        if cfg.use_vote
        else init_filtered
    )
    filtered = _filter_outliers_knn(consensus, cfg.knn_k) if cfg.use_knn else consensus
    logger.info(
        f"  obj {obj_id}: {len(all_pts)} raw -> {len(init_filtered)} init-knn "
        f"-> {len(consensus)} multiview -> {len(filtered)} bbox-knn "
        f"(knn={cfg.use_knn}, vote={cfg.use_vote})"
    )
    return filtered


def _fused_object_box(
    chunks: list[np.ndarray],
    obj_id: int,
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    seg_store: SegmentationStore,
    cfg: ObjectBoxConfig,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, np.ndarray]:
    all_pts = _subsample(
        np.concatenate(chunks, axis=0), obj_id, cfg.max_points_per_object
    )
    logger.info(f"  obj {obj_id}: {len(all_pts)} raw points")
    filtered = _fused_object_points(
        all_pts, obj_id, sequence_id, frame_store, seg_store, cfg
    )
    # Only skip when an OBB is genuinely un-fittable (matches production,
    # which drops objects with <3 points after pruning).
    if len(filtered) < 3:
        logger.info(f"  obj {obj_id}: <3 points, cannot fit OBB, skipping")
        return None, filtered
    return _fit_gravity_aligned_obb(filtered), filtered


def _median_object_box(
    chunks: list[np.ndarray],
    obj_id: int,
    cfg: ObjectBoxConfig,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    frame_chunks = [_subsample(c, obj_id, cfg.max_points_per_object) for c in chunks]
    box, kept = _median_box_from_frames(
        frame_chunks, cfg.knn_k, cfg.use_knn, cfg.median_min_frame_points
    )
    if box is None:
        raise RuntimeError(
            f"obj {obj_id}: no frame has >= {cfg.median_min_frame_points} points "
            f"(per-frame counts: {[len(c) for c in chunks]}); cannot fit a median box"
        )
    points = np.concatenate(kept, axis=0)
    logger.info(
        f"  obj {obj_id}: median box over {len(kept)}/{len(chunks)} valid frames "
        f"({len(points)} points, knn={cfg.use_knn})"
    )
    return box, _subsample(points, obj_id, cfg.max_points_per_object)


def _first_frame_object_box(
    frame_chunks: list[tuple[int, np.ndarray]],
    obj_id: int,
    cfg: ObjectBoxConfig,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    sub = [(t, _subsample(c, obj_id, cfg.max_points_per_object)) for t, c in frame_chunks]
    try:
        box, points, ts, idx = _first_frame_box(
            sub, cfg.knn_k, cfg.use_knn, cfg.median_min_frame_points
        )
    except RuntimeError as e:
        raise RuntimeError(f"obj {obj_id}: {e}") from e
    first_ts = min(t for t, _ in frame_chunks)
    logger.info(
        f"  obj {obj_id}: first valid frame {ts} = mask frame #{idx} of "
        f"{len(frame_chunks)} ({(ts - first_ts) / 1e9:.3f} s after the first "
        f"mask; {len(points)} points, knn={cfg.use_knn})"
    )
    return box, points


def _object_box(
    frame_chunks: list[tuple[int, np.ndarray]],
    obj_id: int,
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    seg_store: SegmentationStore,
    cfg: ObjectBoxConfig,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, np.ndarray]:
    if cfg.box_mode == "first_frame":
        return _first_frame_object_box(frame_chunks, obj_id, cfg)
    chunks = [c for _, c in frame_chunks if len(c) > 0]
    if cfg.box_mode == "median":
        return _median_object_box(chunks, obj_id, cfg)
    return _fused_object_box(chunks, obj_id, sequence_id, frame_store, seg_store, cfg)


def _write_object_reconstruction(
    scene_store: SQLiteSceneStore,
    sequence_id: str,
    obj: SceneObject,
    box: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    obb_transform, obb_aabb, position = box
    recon = ObjectReconstruction(
        reconstruction_id=0,
        sequence_id=sequence_id,
        object_id=obj.object_id,
        time_range_start_ns=obj.first_seen_ns,
        time_range_end_ns=obj.last_seen_ns,
        obb_aabb=obb_aabb,
        obb_transform=obb_transform,
        position=position,
        initial_obb_aabb=obb_aabb,
        initial_obb_transform=obb_transform,
        initial_position=position,
        num_gaussians=None,
        psnr=None,
        ssim=None,
        lpips=None,
        created_ns=int(time.time() * 1e9),
    )
    scene_store.write_reconstruction(recon)
    logger.info(
        f"  obj {obj.object_id} ({obj.query_name}): OBB written, "
        f"position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
    )


def _process_sequence(
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    seg_store: SegmentationStore,
    scene_store: SQLiteSceneStore,
    points_store: SQLiteObjectPointsStore,
    cfg: ObjectBoxConfig,
    depth_stride: int,
    mask_erosion: int = 0,
) -> None:
    logger.info(f"[{sequence_id}] Collecting 3D points from depth maps...")
    t0 = time.monotonic()
    raw_points = _collect_object_points(
        sequence_id, frame_store, seg_store, depth_stride, mask_erosion
    )
    logger.info(
        f"[{sequence_id}] Collected points for {len(raw_points)} objects "
        f"in {time.monotonic() - t0:.1f}s"
    )

    build_scene(sequence_id, seg_store, scene_store, frame_store)
    obj_map = {o.object_id: o for o in scene_store.get_all_scene_objects(sequence_id)}

    recon_count = 0
    for obj_id, frame_chunks in raw_points.items():
        if not any(len(c) > 0 for _, c in frame_chunks):
            continue
        box, points = _object_box(
            frame_chunks, obj_id, sequence_id, frame_store, seg_store, cfg
        )
        points_store.write_points(sequence_id, obj_id, points)
        if box is None:
            continue
        _write_object_reconstruction(scene_store, sequence_id, obj_map[obj_id], box)
        recon_count += 1

    logger.info(f"[{sequence_id}] Wrote {recon_count} reconstructions")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir)

    frames_dir = Path(args.frames_dir)
    frames_db = frames_dir / "frames.db"
    if not frames_db.exists():
        raise RuntimeError(f"frames.db not found at {frames_db}")

    cfg = ObjectBoxConfig(
        knn_k=args.knn_k,
        use_knn=not args.no_knn,
        use_vote=not args.no_multiview_vote,
        box_mode=args.box_mode,
        median_min_frame_points=args.median_min_frame_points,
        max_points_per_object=args.max_points_per_object,
    )
    logger.info(f"Object box config: {cfg}")

    frame_store = SQLiteFrameStore(frames_db, read_only=True)
    logger.info(f"Loading segmentations from HF dataset: {args.dataset}")
    seg_store = load_segmentation_store(args.dataset)
    scene_store = SQLiteSceneStore(output_dir / "scene.db")
    points_store = SQLiteObjectPointsStore(output_dir / "object_points.db")

    sequence_ids = frame_store.get_all_sequence_ids()
    if args.sequence_ids:
        sequence_ids = [s for s in sequence_ids if s in set(args.sequence_ids)]
        missing = set(args.sequence_ids) - set(sequence_ids)
        if missing:
            raise RuntimeError(f"--sequence-ids not in frames.db: {sorted(missing)}")
    logger.info(f"Processing {len(sequence_ids)} sequences")

    for seq_id in sequence_ids:
        _process_sequence(
            seq_id,
            frame_store,
            seg_store,
            scene_store,
            points_store,
            cfg,
            depth_stride=args.depth_stride,
            mask_erosion=args.mask_erosion,
        )

    frame_store.close()
    seg_store.close()
    scene_store.close()
    points_store.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
