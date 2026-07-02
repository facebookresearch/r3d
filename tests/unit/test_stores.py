# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from r3d.pipeline.frame_data import CameraIntrinsics, DepthSource
from r3d.pipeline.stores.base import ObjectReconstruction, SceneObject
from r3d.pipeline.stores.sqlite_store import (
    SQLiteFrameStore,
    SQLiteObjectPointsStore,
    SQLiteSceneStore,
    SQLiteSegmentationStore,
)


class TestSQLiteFrameStore(unittest.TestCase):
    def test_write_and_read_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "frames.db"
            store = SQLiteFrameStore(db_path)
            intrinsics = CameraIntrinsics(
                fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
            )
            T_sd = np.eye(4)
            T_sd[:3, 3] = [1.0, 2.0, 3.0]
            T_dc = np.eye(4)
            gravity = np.array([0.0, -9.81, 0.0])

            store.write_frame(
                sequence_id="seq001",
                timestamp_ns=100,
                rgb_path="frame_data/rgb.png",
                depth_path="frame_data/depth.png",
                intrinsics=intrinsics,
                T_scene_device=T_sd,
                T_device_camera=T_dc,
                depth_source=DepthSource.GROUND_TRUTH,
                gravity_world=gravity,
            )

            timestamps = store.get_all_timestamps("seq001")
            self.assertEqual(timestamps, [100])
            self.assertEqual(store.get_frame_count("seq001"), 1)

            seq_ids = store.get_all_sequence_ids()
            self.assertEqual(seq_ids, ["seq001"])

            store.close()

    def test_multiple_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "frames.db"
            store = SQLiteFrameStore(db_path)
            intrinsics = CameraIntrinsics(
                fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100
            )
            for ts in [100, 200, 300]:
                store.write_frame(
                    sequence_id="seq001",
                    timestamp_ns=ts,
                    rgb_path=f"frame_data/{ts}_rgb.png",
                    depth_path=f"frame_data/{ts}_depth.png",
                    intrinsics=intrinsics,
                    T_scene_device=np.eye(4),
                    T_device_camera=np.eye(4),
                    depth_source=DepthSource.GROUND_TRUTH,
                )

            self.assertEqual(store.get_frame_count("seq001"), 3)
            timestamps = store.get_all_timestamps("seq001")
            self.assertEqual(timestamps, [100, 200, 300])
            store.close()


class TestSQLiteObjectPointsStore(unittest.TestCase):
    def test_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "object_points.db"
            store = SQLiteObjectPointsStore(db_path)

            points = np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                dtype=np.float32,
            )
            store.write_points("seq001", object_id=42, points=points)

            loaded = store.get_points("seq001", object_id=42)
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded, points, atol=1e-6)
            store.close()

    def test_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "object_points.db"
            store = SQLiteObjectPointsStore(db_path)
            self.assertIsNone(store.get_points("seq001", object_id=99))
            store.close()

    def test_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "object_points.db"
            store = SQLiteObjectPointsStore(db_path)

            pts1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            pts2 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

            store.write_points("seq001", 1, pts1)
            store.write_points("seq001", 1, pts2)

            loaded = store.get_points("seq001", 1)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (2, 3))
            np.testing.assert_allclose(loaded, pts2, atol=1e-6)
            store.close()

    def test_rejects_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "object_points.db"
            store = SQLiteObjectPointsStore(db_path)
            with self.assertRaises(ValueError):
                store.write_points("seq001", 1, np.array([1, 2, 3], dtype=np.float32))
            store.close()


class TestSQLiteSceneStore(unittest.TestCase):
    def _make_reconstruction(
        self, seq_id: str = "seq001", obj_id: int = 1
    ) -> ObjectReconstruction:
        return ObjectReconstruction(
            reconstruction_id=0,
            sequence_id=seq_id,
            object_id=obj_id,
            time_range_start_ns=100,
            time_range_end_ns=500,
            obb_aabb=np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]),
            obb_transform=np.eye(4),
            position=np.array([0.5, 1.0, 1.5]),
            initial_obb_aabb=np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]),
            initial_obb_transform=np.eye(4),
            initial_position=np.array([0.5, 1.0, 1.5]),
            psnr=25.0,
            ssim=0.9,
            lpips=0.1,
            created_ns=1_000_000,
        )

    def test_scene_object_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "scene.db"
            store = SQLiteSceneStore(db_path)

            obj = SceneObject(
                sequence_id="seq001",
                object_id=1,
                query_name="coffee mug",
                first_seen_ns=100,
                last_seen_ns=500,
            )
            store.write_scene_object(obj)
            objects = store.get_all_scene_objects("seq001")
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0].query_name, "coffee mug")
            self.assertEqual(objects[0].object_id, 1)
            store.close()

    def test_reconstruction_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "scene.db"
            store = SQLiteSceneStore(db_path)

            recon = self._make_reconstruction()
            store.write_reconstruction(recon)

            recons = store.get_reconstructions("seq001", 1)
            self.assertEqual(len(recons), 1)
            loaded = recons[0]
            np.testing.assert_allclose(
                loaded.obb_aabb,
                np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]),
                atol=1e-10,
            )
            np.testing.assert_allclose(loaded.obb_transform, np.eye(4), atol=1e-10)
            np.testing.assert_allclose(
                loaded.position, np.array([0.5, 1.0, 1.5]), atol=1e-10
            )
            self.assertAlmostEqual(loaded.psnr, 25.0)
            self.assertAlmostEqual(loaded.ssim, 0.9)
            store.close()

    def test_reconstruction_without_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "scene.db"
            store = SQLiteSceneStore(db_path)

            recon = ObjectReconstruction(
                reconstruction_id=0,
                sequence_id="seq001",
                object_id=2,
                time_range_start_ns=100,
                time_range_end_ns=500,
                obb_aabb=np.zeros(6),
                obb_transform=np.eye(4),
                position=np.zeros(3),
                initial_obb_aabb=np.zeros(6),
                initial_obb_transform=np.eye(4),
                initial_position=np.zeros(3),
                num_gaussians=None,
                psnr=None,
                ssim=None,
                lpips=None,
                created_ns=1_000_000,
            )
            store.write_reconstruction(recon)
            recons = store.get_reconstructions("seq001", 2)
            self.assertEqual(len(recons), 1)
            self.assertIsNone(recons[0].psnr)
            self.assertIsNone(recons[0].num_gaussians)
            store.close()

    def test_is_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "scene.db"
            store = SQLiteSceneStore(db_path)
            self.assertFalse(store.is_non_empty())
            store.write_scene_object(
                SceneObject(
                    sequence_id="seq001",
                    object_id=1,
                    query_name="mug",
                    first_seen_ns=0,
                    last_seen_ns=100,
                )
            )
            self.assertTrue(store.is_non_empty())
            store.close()


class TestSQLiteSegmentationStore(unittest.TestCase):
    def test_register_and_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seg.db"
            store = SQLiteSegmentationStore(db_path)

            store.register_query("seq001", "knife", [100, 200, 300])
            pending = store.get_pending_timestamps("seq001", "knife")
            self.assertEqual(pending, [100, 200, 300])

            store.mark_segmented("seq001", 100, "knife")
            pending = store.get_pending_timestamps("seq001", "knife")
            self.assertEqual(pending, [200, 300])

            segmented = store.get_segmented_timestamps("seq001", "knife")
            self.assertEqual(segmented, [100])

            store.close()

    def test_query_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "seg.db"
            store = SQLiteSegmentationStore(db_path)
            store.register_query("seq001", "fork", [100])
            store.register_query("seq001", "knife", [100])
            names = store.get_all_query_names("seq001")
            self.assertEqual(sorted(names), ["fork", "knife"])
            store.close()


if __name__ == "__main__":
    unittest.main()
