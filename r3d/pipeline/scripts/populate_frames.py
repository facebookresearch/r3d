# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Populate frames.db from ADT sequences.

Loads ADT sequences from local paths, extracts RGB + depth frames at the
requested FPS, and writes them into frames.db via SQLiteFrameStore.

Supports parallel extraction via --num-workers. Workers write PNG files
directly to frame_data/, then the main process builds frames.db from
the written files and per-sequence metadata.

Usage:
    python -m r3d.pipeline.scripts.populate_frames \
      --sequence-paths /data/adt/seq131 /data/adt/seq133 \
      --output-dir /tmp/eval \
      --fps 3 \
      --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
from projectaria_tools.core.stream_id import StreamId  # pyre-ignore[21]
from projectaria_tools.utils.calibration_utils import (  # pyre-ignore[21]
    rotate_upright_image_and_calibration,
    undistort_image_and_calibration,
)
from r3d.pipeline.frame_data import CameraIntrinsics, DepthSource
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore
from r3d.utils.adt_provider import load_adt_provider
from r3d.utils.aria_images import get_corrected_rgb_with_calibration
from r3d.utils.logging import setup_logging

logger: logging.Logger = logging.getLogger(__name__)

_UINT16_MAX_MM: int = 65535


def _extract_intrinsics(calibration: object) -> CameraIntrinsics:
    focal = calibration.get_focal_lengths()  # pyre-ignore[16]
    pp = calibration.get_principal_point()  # pyre-ignore[16]
    img_size = calibration.get_image_size()  # pyre-ignore[16]
    return CameraIntrinsics(
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=float(pp[0]),
        cy=float(pp[1]),
        width=int(img_size[0]),
        height=int(img_size[1]),
    )


def _subsample_timestamps(
    timestamps: list[int], target_fps: float, source_fps: float = 30.0
) -> list[int]:
    if target_fps >= source_fps or target_fps <= 0:
        return timestamps
    step = max(1, int(round(source_fps / target_fps)))
    return timestamps[::step]


def _sequence_id_from_path(sequence_path: str) -> str:
    return sequence_path.rstrip("/").split("/")[-1]


def _load_depth_map(
    provider: object,
    ts: int,
    stream_id: StreamId,
) -> np.ndarray:
    depth_with_dt = provider.get_depth_image_by_timestamp_ns(
        ts, stream_id
    )  # pyre-ignore[16]
    if not depth_with_dt.is_valid():
        raise RuntimeError(f"Depth unavailable at timestamp {ts}")
    depth_raw = depth_with_dt.data().to_numpy_array().astype(np.float32) * 0.001
    camera_calib_raw = provider.get_aria_camera_calibration(
        stream_id
    )  # pyre-ignore[16]
    if camera_calib_raw is None:
        raise RuntimeError(
            f"Camera calibration unavailable for depth at timestamp {ts}"
        )
    depth_map, _depth_calib = undistort_image_and_calibration(
        depth_raw, camera_calib_raw
    )
    depth_map, _depth_calib = rotate_upright_image_and_calibration(
        depth_map, _depth_calib
    )
    return depth_map


def _resize_frame(
    rgb: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    orig_h, orig_w = rgb.shape[:2]
    rgb = cv2.resize(rgb, (image_size, image_size))
    depth_map = cv2.resize(
        depth_map, (image_size, image_size), interpolation=cv2.INTER_NEAREST
    )
    scale_x = image_size / orig_w
    scale_y = image_size / orig_h
    intrinsics = CameraIntrinsics(
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
        width=image_size,
        height=image_size,
    )
    return rgb, depth_map, intrinsics


def _depth_map_to_uint16_mm(depth_map: np.ndarray) -> np.ndarray:
    if not np.isfinite(depth_map).all():
        raise ValueError("depth_map contains NaN or inf values")
    depth_mm = depth_map * 1000.0
    if depth_mm.min() < 0 or depth_mm.max() > _UINT16_MAX_MM:
        raise ValueError(
            f"depth_map values out of uint16 mm range "
            f"[0, {_UINT16_MAX_MM}], got [{depth_mm.min()}, {depth_mm.max()}]"
        )
    return np.round(depth_mm).astype(np.uint16)


def _extract_sequence(
    sequence_path: str,
    output_dir: str,
    fps: float,
    max_frames: int,
    image_size: int | None,
) -> str:
    """Extract frames from one sequence, writing PNGs directly to frame_data/.

    Also writes a per-sequence metadata JSON for the consolidation phase.
    """
    sequence_id = _sequence_id_from_path(sequence_path)
    provider, _data_paths = load_adt_provider(sequence_path)
    stream_id = StreamId("214-1")

    all_timestamps = provider.get_aria_device_capture_timestamps_ns(stream_id)
    gt_start = provider.get_start_time_ns()
    gt_end = provider.get_end_time_ns()
    timestamps = [t for t in all_timestamps if gt_start <= t <= gt_end]
    timestamps = _subsample_timestamps(timestamps, fps)
    if max_frames > 0:
        timestamps = timestamps[:max_frames]

    frame_data_dir = os.path.join(output_dir, "frame_data")
    os.makedirs(frame_data_dir, exist_ok=True)

    gravity_world: np.ndarray | None = None
    T_device_camera: np.ndarray | None = None
    frame_metas: list[dict] = []

    for i, ts in enumerate(timestamps):
        rgb, calibration = get_corrected_rgb_with_calibration(provider, ts, stream_id)
        if rgb is None or calibration is None:
            raise RuntimeError(f"RGB/calibration unavailable at {ts}")

        depth_map = _load_depth_map(provider, ts, stream_id)

        pose_result = provider.get_aria_3d_pose_by_timestamp_ns(ts)  # pyre-ignore[16]
        if not pose_result.is_valid():
            raise RuntimeError(f"Pose invalid at {ts}")

        T_scene_device = pose_result.data().transform_scene_device.to_matrix()
        intrinsics = _extract_intrinsics(calibration)

        if image_size is not None:
            rgb, depth_map, intrinsics = _resize_frame(
                rgb, depth_map, intrinsics, image_size
            )

        if T_device_camera is None:
            T_device_camera = calibration.get_transform_device_camera().to_matrix()
        if gravity_world is None:
            gravity_world = np.array(pose_result.data().gravity_world, dtype=np.float64)

        prefix = f"{sequence_id}_{ts}"
        rgb_rel = f"frame_data/{prefix}_rgb.png"
        depth_rel = f"frame_data/{prefix}_depth.png"
        rgb_path = os.path.join(output_dir, rgb_rel)
        depth_path = os.path.join(output_dir, depth_rel)

        if not cv2.imwrite(
            rgb_path,
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise RuntimeError(f"Failed to write RGB: {rgb_path}")

        depth_mm = _depth_map_to_uint16_mm(depth_map)
        if not cv2.imwrite(depth_path, depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise RuntimeError(f"Failed to write depth: {depth_path}")

        frame_metas.append(
            {
                "timestamp_ns": ts,
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "width": intrinsics.width,
                "height": intrinsics.height,
                "T_scene_device": T_scene_device.tolist(),
            }
        )

    meta_path = os.path.join(output_dir, f".{sequence_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "sequence_id": sequence_id,
                "T_device_camera": T_device_camera.tolist(),
                "gravity_world": gravity_world.tolist(),
                "frames": frame_metas,
            },
            f,
        )

    print(f"[{sequence_id}] Extracted {len(timestamps)} frames", flush=True)
    return sequence_id


def _extract_worker(args: tuple) -> str:
    return _extract_sequence(*args)


def _build_db(output_dir: Path) -> int:
    """Build frames.db from per-sequence metadata JSON files."""
    store = SQLiteFrameStore(output_dir / "frames.db")
    total = 0

    meta_files = sorted(
        f
        for f in os.listdir(output_dir)
        if f.startswith(".") and f.endswith("_meta.json")
    )
    for meta_file in meta_files:
        with open(output_dir / meta_file) as f:
            seq_meta = json.load(f)

        sequence_id = seq_meta["sequence_id"]
        T_device_camera = np.array(seq_meta["T_device_camera"], dtype=np.float64)
        gravity_world = np.array(seq_meta["gravity_world"], dtype=np.float64)

        for fm in seq_meta["frames"]:
            store.write_frame(
                sequence_id=sequence_id,
                timestamp_ns=fm["timestamp_ns"],
                rgb_path=fm["rgb_path"],
                depth_path=fm["depth_path"],
                intrinsics=CameraIntrinsics(
                    fx=fm["fx"],
                    fy=fm["fy"],
                    cx=fm["cx"],
                    cy=fm["cy"],
                    width=fm["width"],
                    height=fm["height"],
                ),
                T_scene_device=np.array(fm["T_scene_device"], dtype=np.float64),
                T_device_camera=T_device_camera,
                gravity_world=gravity_world,
                depth_source=DepthSource.GROUND_TRUTH,
            )
            total += 1

        logger.info(
            f"[{sequence_id}] Wrote {len(seq_meta['frames'])} rows to frames.db"
        )
        os.remove(output_dir / meta_file)

    store.close()
    return total


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate frames.db from ADT sequences"
    )
    parser.add_argument(
        "--sequence-paths",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for frame extraction (default: 1).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=args.output_dir)

    seen_ids: set[str] = set()
    for seq_path in args.sequence_paths:
        seq_id = _sequence_id_from_path(seq_path)
        if seq_id in seen_ids:
            raise RuntimeError(f"Duplicate sequence_id '{seq_id}'")
        seen_ids.add(seq_id)

    worker_args = [
        (seq_path, str(args.output_dir), args.fps, args.max_frames, args.image_size)
        for seq_path in args.sequence_paths
    ]

    num_workers = min(args.num_workers, len(worker_args))
    if num_workers <= 1:
        for wa in worker_args:
            _extract_sequence(*wa)
    else:
        logger.info(
            f"Extracting {len(worker_args)} sequences with {num_workers} workers"
        )
        with mp.Pool(num_workers) as pool:
            pool.map(_extract_worker, worker_args)

    logger.info("=== Building frames.db ===")
    total = _build_db(args.output_dir)
    logger.info(f"Total: {total} frames written to frames.db")


if __name__ == "__main__":
    main()
