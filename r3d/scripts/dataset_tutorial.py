# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tutorial: how to use the R3D-Bench dataset.

Runnable demos (choose with --demos, default: all). Each reuses the same
pipeline utilities the eval uses (HF parquet stores, depth-lifting + OBB fit
from build_scene, mesh rescaling from volume, viz/video helpers):

  annotation   : load a QA annotation and print its fields.
  mesh         : load a SAM3D mesh (embedded GLB) and save it to disk.
  pointing     : overlay a referenced object's SAM3 mask on its reference frame.
  scene_video  : overlay all SAM3 masks across a sequence into an MP4 with the
                 question + GT answer in a caption bar.
  depth_lift   : depth-lift one volumetric object (default: a mug) and save its
                 3D point cloud, mask, fitted 3D bbox (drawn), and rescaled mesh.

Usage:
    python -m r3d.scripts.dataset_tutorial \
        --dataset facebook/r3d-bench --frames-dir $ASSETS/frames \
        --output-dir /tmp/r3d_tutorial
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from projectaria_tools.core.stream_id import StreamId
from r3d.data_gen.generators.disambiguation import _extract_object_mask
from r3d.data_gen.utils.annotation_schema import DisambiguationMethod
from r3d.data_gen.utils.sequence import load_sequence_data
from r3d.pipeline.eval.config import DEFAULT_DATASET
from r3d.pipeline.hf_dataset import (
    load_annotation_store,
    load_mesh_store,
    load_segmentation_store,
)
from r3d.pipeline.scripts.build_scene import (
    _collect_object_points,
    _filter_outliers_knn,
    _fit_gravity_aligned_obb,
    _multiview_consensus_filter,
)
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore
from r3d.pipeline.volume import rescale_mesh_to_obb
from r3d.utils.logging import setup_logging
from r3d.utils.video import write_video
from r3d.utils.viz import (
    draw_bbox_from_mask,
    draw_wireframe,
    overlay_mask,
    project_obb_to_2d,
    track_color,
)

logger: logging.Logger = logging.getLogger(__name__)

# RGB camera stream used throughout the ADT recordings.
DEFAULT_STREAM_ID = StreamId("214-1")

KNN_K = 6


def _write_ply(path: Path, points: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    logger.info(f"  wrote {path} ({len(points)} points)")


def _add_caption(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Add a black caption bar with ASCII text at the bottom of an RGB frame."""
    w = frame.shape[1]
    bar_h = 22 * len(lines) + 12
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            bar,
            line[:110],
            (8, 22 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([frame, bar])


def _find_object_id(seg_store, sequence_id: str, object_name: str) -> int | None:
    """Map a canonical object name to its SAM3 object_id in a sequence."""
    for ts in seg_store.get_segmented_timestamps(sequence_id, object_name):
        objs = seg_store.get_segmentation(sequence_id, ts, object_name).objects
        if objs:
            return next(iter(objs))
    return None


def demo_annotation(ann_store, sequence_id: str) -> None:
    anns = ann_store.get_annotations_by_sequence(sequence_id)
    logger.info(f"[annotation] {len(anns)} annotations in {sequence_id}")
    a = anns[0]
    logger.info(f"  id={a.annotation_id}")
    logger.info(f"  type={a.query_layer.question_type.value}")
    logger.info(f"  Q: {a.query_layer.question_text}")
    logger.info(f"  GT: {a.query_layer.gt_answer} ({a.query_layer.gt_answer_type})")
    logger.info(
        f"  referenced objects: "
        f"{[o.canonical_name for o in a.identity_layer.referenced_objects]}"
    )


def demo_mesh(mesh_store, sequence_id: str, object_name: str, out: Path) -> None:
    mesh = mesh_store.get_mesh(sequence_id, object_name)
    if mesh is None:
        logger.info(f"[mesh] no mesh for ({sequence_id}, {object_name})")
        return
    dst = out / f"mesh_{object_name.replace(' ', '_')}.glb"
    dst.write_bytes(
        Path(mesh_store.get_mesh_abs_path(sequence_id, object_name)).read_bytes()
    )
    logger.info(
        f"[mesh] {object_name}: {mesh.num_vertices} verts, {mesh.num_faces} faces -> {dst}"
    )


def demo_pointing(
    ann_store, frame_store, sequence_id: str, adt_root: str, out: Path
) -> None:
    """Visualize a pointing annotation.

    A pointing annotation designates its target object on a single reference
    frame (one per object per video), via the object's ``reference_timestamp_ns``
    and ``adt_instance_id``. The pointing mask is not stored -- it is the ADT
    ground-truth segmentation of that instance at that reference frame, derived
    here from the ADT provider and overlaid on the pipeline's frame.
    """
    # Find a pointing-disambiguated object with a reference frame.
    target = None
    for ann in ann_store.get_annotations_by_sequence(sequence_id):
        if ann.disambiguation_layer.method != DisambiguationMethod.POINTING:
            continue
        for ref_obj in ann.identity_layer.referenced_objects:
            if ref_obj.reference_timestamp_ns is not None:
                target = ref_obj
                break
        if target is not None:
            break
    if target is None:
        logger.info(f"[pointing] no pointing annotation with a reference frame in {sequence_id}")
        return

    gt_provider, _ = load_sequence_data(str(Path(adt_root) / sequence_id), verbose=False)
    ts = target.reference_timestamp_ns
    mask = _extract_object_mask(
        gt_provider, target.adt_instance_id, ts, DEFAULT_STREAM_ID
    )
    if mask is None or not mask.any():
        logger.info(f"[pointing] no ADT mask for {target.canonical_name} at ts={ts}")
        return

    frame = frame_store.load_frame(sequence_id, ts).rgb.copy()
    if mask.shape[:2] != frame.shape[:2]:
        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        )

    color = track_color(0)
    overlay_mask(frame, mask, color, inplace=True)
    draw_bbox_from_mask(frame, mask, color, target.canonical_name, inplace=True)
    dst = out / f"pointing_{target.canonical_name.replace(' ', '_')}.png"
    cv2.imwrite(str(dst), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    logger.info(
        f"[pointing] {target.canonical_name}: ADT GT mask at reference "
        f"ts={ts} -> {dst}"
    )


def demo_scene_video(
    ann_store, seg_store, frame_store, sequence_id: str, out: Path
) -> None:
    anns = ann_store.get_annotations_by_sequence(sequence_id)
    caption = [
        f"Q: {anns[0].query_layer.question_text}",
        f"GT: {anns[0].query_layer.gt_answer}",
    ]
    query_names = seg_store.get_all_query_names(sequence_id)
    timestamps = frame_store.get_all_timestamps(sequence_id)
    frames_out: list[np.ndarray] = []
    for ts in timestamps:
        frame = frame_store.load_frame(sequence_id, ts).rgb.copy()
        for qn in query_names:
            seg = seg_store.get_segmentation(sequence_id, ts, qn)
            for obj in seg.objects.values():
                color = track_color(obj.object_id)
                overlay_mask(frame, obj.mask, color, inplace=True)
                draw_bbox_from_mask(frame, obj.mask, color, qn, inplace=True)
        frames_out.append(_add_caption(frame, caption))
    # Derive playback FPS from the frame timestamps (ns) so the video plays at
    # real time (frames are extracted at ~3 FPS), not a fixed guessed rate.
    fps = 1e9 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 3.0
    dst = out / f"scene_{sequence_id}.mp4"
    write_video(frames_out, dst, fps=fps)
    logger.info(f"[scene_video] {len(frames_out)} frames @ {fps:.2f} fps -> {dst}")


def demo_depth_lift(
    frame_store, seg_store, mesh_store, sequence_id: str, object_name: str, out: Path
) -> None:
    obj_id = _find_object_id(seg_store, sequence_id, object_name)
    if obj_id is None:
        logger.info(f"[depth_lift] object '{object_name}' not found in {sequence_id}")
        return
    # Reuse the exact build_scene recipe: lift -> KNN -> multiview vote -> KNN -> OBB.
    raw = _collect_object_points(sequence_id, frame_store, seg_store, 4, 0).get(obj_id)
    if not raw:
        logger.info(f"[depth_lift] no points for {object_name}")
        return
    pts = np.concatenate(raw, axis=0)
    pts = _filter_outliers_knn(pts, KNN_K)
    pts = _multiview_consensus_filter(pts, obj_id, sequence_id, frame_store, seg_store)
    pts = _filter_outliers_knn(pts, KNN_K)
    result = _fit_gravity_aligned_obb(pts)
    if result is None:
        logger.info(f"[depth_lift] could not fit OBB for {object_name}")
        return
    transform, aabb, _ = result
    tag = object_name.replace(" ", "_")

    _write_ply(out / f"points_{tag}.ply", pts)
    np.savez(out / f"bbox_{tag}.npz", obb_aabb=aabb, obb_transform=transform)

    # Save the object's mask + the 3D bbox drawn on a representative frame.
    ts_list = seg_store.get_segmented_timestamps(sequence_id, object_name)
    ts = ts_list[len(ts_list) // 2]
    fd = frame_store.load_frame(sequence_id, ts)
    obj = next(
        iter(seg_store.get_segmentation(sequence_id, ts, object_name).objects.values())
    )
    cv2.imwrite(str(out / f"mask_{tag}.png"), obj.mask.astype(np.uint8) * 255)
    frame = fd.rgb.copy()
    corners = project_obb_to_2d(
        aabb,
        transform,
        fd.T_scene_device,
        fd.T_device_camera,
        fd.intrinsics.fx,
        fd.intrinsics.fy,
        fd.intrinsics.cx,
        fd.intrinsics.cy,
    )
    if corners is not None:
        draw_wireframe(frame, corners, track_color(obj_id), thickness=2)
    cv2.imwrite(
        str(out / f"bbox_render_{tag}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    )

    # Rescale the SAM3D mesh to the fitted OBB and save it.
    mesh_path = mesh_store.get_mesh_abs_path(sequence_id, object_name)
    if mesh_path is not None:
        rescaled = rescale_mesh_to_obb(mesh_path, aabb)
        if rescaled is not None:
            rescaled.export(str(out / f"rescaled_mesh_{tag}.glb"))
            logger.info(f"  wrote rescaled mesh for {object_name}")
    logger.info(
        f"[depth_lift] {object_name}: {len(pts)} points, saved artifacts to {out}"
    )


ALL_DEMOS = ["annotation", "mesh", "pointing", "scene_video", "depth_lift"]


def main() -> None:
    p = argparse.ArgumentParser(description="R3D-Bench dataset usage tutorial")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--frames-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--sequence", type=str, default="Apartment_release_clean_seq131_M1292"
    )
    p.add_argument("--object", type=str, default="mug")
    p.add_argument(
        "--adt-root",
        type=str,
        default=None,
        help="ADT sequences directory. Required for the 'pointing' demo, which "
        "derives the pointing mask from the ADT ground-truth segmentation.",
    )
    p.add_argument("--demos", type=str, nargs="+", default=ALL_DEMOS, choices=ALL_DEMOS)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=out)

    frames_db = Path(args.frames_dir) / "frames.db"
    if not frames_db.exists():
        raise RuntimeError(f"frames.db not found at {frames_db}")
    frame_store = SQLiteFrameStore(frames_db, read_only=True)
    ann_store = load_annotation_store(args.dataset)
    seg_store = load_segmentation_store(args.dataset)
    mesh_store = load_mesh_store(args.dataset)

    if "annotation" in args.demos:
        demo_annotation(ann_store, args.sequence)
    if "mesh" in args.demos:
        demo_mesh(mesh_store, args.sequence, args.object, out)
    if "pointing" in args.demos:
        if args.adt_root is None:
            logger.info("[pointing] skipped: pass --adt-root to run this demo")
        else:
            demo_pointing(ann_store, frame_store, args.sequence, args.adt_root, out)
    if "scene_video" in args.demos:
        demo_scene_video(ann_store, seg_store, frame_store, args.sequence, out)
    if "depth_lift" in args.demos:
        demo_depth_lift(
            frame_store, seg_store, mesh_store, args.sequence, args.object, out
        )


if __name__ == "__main__":
    main()
