# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Build scene.db from frames.db and the R3D-Bench segmentations.

Replaces the internal orchestrator + gsplat + scene_builder chain with a
simpler depth-lifting pipeline:

1. Load frames from frames.db and segmentations from the HF dataset
2. For each segmented object across all frames:
   - Depth-lift masked pixels to 3D world coordinates
3. KNN outlier filtering on the aggregated 3D points
4. Fit gravity-aligned oriented bounding boxes (OBB)
5. Write per-object 3D points, scene objects, reconstructions, frame
   visibility, and coverage to scene.db

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
from pathlib import Path

import cv2
import numpy as np
from r3d.pipeline.eval.config import DEFAULT_DATASET
from r3d.pipeline.frame_data import CameraIntrinsics
from r3d.pipeline.scene_builder import build_scene
from r3d.pipeline.stores.base import ObjectReconstruction, SegmentationStore
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
    return parser


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
) -> dict[int, list[np.ndarray]]:
    timestamps = frame_store.get_all_timestamps(sequence_id)
    query_names = seg_store.get_all_query_names(sequence_id)
    object_points: dict[int, list[np.ndarray]] = {}
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
                if len(pts) > 0:
                    object_points.setdefault(obj_id, []).append(pts)

    return object_points


def _process_sequence(
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    seg_store: SegmentationStore,
    scene_store: SQLiteSceneStore,
    points_store: SQLiteObjectPointsStore,
    knn_k: int,
    min_points: int,
    depth_stride: int,
    mask_erosion: int = 0,
    max_points_per_object: int = 50000,
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

    objects = scene_store.get_all_scene_objects(sequence_id)
    obj_map = {obj.object_id: obj for obj in objects}

    recon_count = 0
    for obj_id, chunks in raw_points.items():
        all_pts = np.concatenate(chunks, axis=0)
        if len(all_pts) > max_points_per_object:
            rng = np.random.RandomState(obj_id)
            indices = rng.choice(len(all_pts), max_points_per_object, replace=False)
            all_pts = all_pts[indices]
        logger.info(f"  obj {obj_id}: {len(all_pts)} raw points")

        # Production recipe: init-KNN (prune_init_knn) -> multiview membership
        # vote (identity argmax) -> bbox-KNN (bbox_method="knn") -> OBB fit.
        init_filtered = _filter_outliers_knn(all_pts, knn_k)
        consensus = _multiview_consensus_filter(
            init_filtered, obj_id, sequence_id, frame_store, seg_store
        )
        filtered = _filter_outliers_knn(consensus, knn_k)
        logger.info(
            f"  obj {obj_id}: {len(all_pts)} raw -> {len(init_filtered)} init-knn "
            f"-> {len(consensus)} multiview -> {len(filtered)} bbox-knn"
        )

        points_store.write_points(sequence_id, obj_id, filtered)

        # Only skip when an OBB is genuinely un-fittable (matches production,
        # which drops objects with <3 points after pruning).
        if len(filtered) < 3:
            logger.info(f"  obj {obj_id}: <3 points, cannot fit OBB, skipping")
            continue

        result = _fit_gravity_aligned_obb(filtered)
        if result is None:
            continue

        obb_transform, obb_aabb, position = result
        obj = obj_map[obj_id]

        recon = ObjectReconstruction(
            reconstruction_id=0,
            sequence_id=sequence_id,
            object_id=obj_id,
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
        recon_count += 1
        logger.info(
            f"  obj {obj_id} ({obj.query_name}): OBB written, "
            f"position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        )

    logger.info(f"[{sequence_id}] Wrote {recon_count} reconstructions")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir)

    frames_dir = Path(args.frames_dir)
    frames_db = frames_dir / "frames.db"
    if not frames_db.exists():
        raise RuntimeError(f"frames.db not found at {frames_db}")

    frame_store = SQLiteFrameStore(frames_db, read_only=True)
    from r3d.pipeline.hf_dataset import load_segmentation_store

    logger.info(f"Loading segmentations from HF dataset: {args.dataset}")
    seg_store = load_segmentation_store(args.dataset)
    scene_store = SQLiteSceneStore(output_dir / "scene.db")
    points_store = SQLiteObjectPointsStore(output_dir / "object_points.db")

    sequence_ids = frame_store.get_all_sequence_ids()
    logger.info(f"Found {len(sequence_ids)} sequences in frames.db")

    for seq_id in sequence_ids:
        _process_sequence(
            seq_id,
            frame_store,
            seg_store,
            scene_store,
            points_store,
            knn_k=args.knn_k,
            min_points=args.min_points,
            depth_stride=args.depth_stride,
            mask_erosion=args.mask_erosion,
            max_points_per_object=args.max_points_per_object,
        )

    frame_store.close()
    seg_store.close()
    scene_store.close()
    points_store.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
