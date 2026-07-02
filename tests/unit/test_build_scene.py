# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

import numpy as np
from r3d.pipeline.frame_data import CameraIntrinsics
from r3d.pipeline.scripts.build_scene import (
    _filter_outliers_knn,
    _fit_gravity_aligned_obb,
    _lift_depth_to_3d,
)


class TestLiftDepthTo3D(unittest.TestCase):
    def test_identity_pose_center_pixel(self) -> None:
        """A pixel at (cx, cy) with depth d should map to (0, 0, d) in cam frame.
        With identity poses, world == cam, so world point is (0, 0, d)."""
        h, w = 100, 100
        depth = np.full((h, w), 5.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        mask[50, 50] = True
        intrinsics = CameraIntrinsics(
            fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=w, height=h
        )
        T_scene_device = np.eye(4)
        T_device_camera = np.eye(4)
        pts = _lift_depth_to_3d(
            depth, mask, intrinsics, T_scene_device, T_device_camera, stride=1
        )
        self.assertEqual(pts.shape[0], 1)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 5.0], atol=1e-6)

    def test_off_center_pixel(self) -> None:
        """Pixel at (u=60, v=50) with depth 5, fx=100, cx=50 should give x_cam=(60-50)/100*5=0.5."""
        h, w = 100, 100
        depth = np.full((h, w), 5.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        mask[50, 60] = True
        intrinsics = CameraIntrinsics(
            fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=w, height=h
        )
        pts = _lift_depth_to_3d(depth, mask, intrinsics, np.eye(4), np.eye(4), stride=1)
        self.assertEqual(pts.shape[0], 1)
        self.assertAlmostEqual(pts[0, 0], 0.5, places=5)
        self.assertAlmostEqual(pts[0, 1], 0.0, places=5)
        self.assertAlmostEqual(pts[0, 2], 5.0, places=5)

    def test_stride(self) -> None:
        """With stride=2 on a 100x100 image, we get a 50x50 grid = 2500 subsampled pixels."""
        h, w = 100, 100
        depth = np.full((h, w), 3.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)
        intrinsics = CameraIntrinsics(
            fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=w, height=h
        )
        pts = _lift_depth_to_3d(depth, mask, intrinsics, np.eye(4), np.eye(4), stride=2)
        self.assertEqual(pts.shape[0], 50 * 50)

    def test_empty_mask(self) -> None:
        """All-false mask should return empty array."""
        h, w = 10, 10
        depth = np.full((h, w), 1.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        intrinsics = CameraIntrinsics(
            fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=w, height=h
        )
        pts = _lift_depth_to_3d(depth, mask, intrinsics, np.eye(4), np.eye(4), stride=1)
        self.assertEqual(pts.shape, (0, 3))

    def test_zero_depth_filtered(self) -> None:
        """Pixels with depth=0 should be excluded even if mask is True."""
        h, w = 10, 10
        depth = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)
        intrinsics = CameraIntrinsics(
            fx=10.0, fy=10.0, cx=5.0, cy=5.0, width=w, height=h
        )
        pts = _lift_depth_to_3d(depth, mask, intrinsics, np.eye(4), np.eye(4), stride=1)
        self.assertEqual(pts.shape[0], 0)


class TestFilterOutliersKnn(unittest.TestCase):
    def test_removes_outlier(self) -> None:
        """Clustered points + 1 far outlier -> outlier removed, cluster mostly kept."""
        grid = np.array(
            [[i, j, 0.0] for i in range(5) for j in range(5)], dtype=np.float64
        )
        outlier = np.array([[1000.0, 1000.0, 1000.0]])
        points = np.vstack([grid, outlier])
        filtered = _filter_outliers_knn(points, k=5)
        self.assertLess(len(filtered), len(points))
        max_dist = np.max(np.linalg.norm(filtered, axis=1))
        self.assertLess(max_dist, 100.0)

    def test_no_outliers_keeps_most(self) -> None:
        """Gaussian cluster: filter should keep the vast majority."""
        rng = np.random.RandomState(0)
        points = rng.randn(100, 3) * 0.01 + np.array([1.0, 2.0, 3.0])
        filtered = _filter_outliers_knn(points, k=5)
        self.assertGreater(len(filtered), 85)

    def test_few_points_returned_as_is(self) -> None:
        """If N <= k, return all points unchanged."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        filtered = _filter_outliers_knn(points, k=5)
        self.assertEqual(len(filtered), 3)


class TestFitGravityAlignedOBB(unittest.TestCase):
    def test_axis_aligned_box(self) -> None:
        """Points forming an axis-aligned box should produce OBB with matching extents."""
        points = np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 3, 0],
                [0, 0, 4],
                [2, 3, 0],
                [2, 0, 4],
                [0, 3, 4],
                [2, 3, 4],
            ],
            dtype=np.float64,
        )
        result = _fit_gravity_aligned_obb(points)
        self.assertIsNotNone(result)
        obb_transform, obb_aabb, position = result

        # Position is centroid
        np.testing.assert_allclose(position, [1.0, 1.5, 2.0], atol=1e-10)

        # OBB extents (in local frame) should cover the box dimensions
        x_extent = obb_aabb[1] - obb_aabb[0]
        y_extent = obb_aabb[3] - obb_aabb[2]
        z_extent = obb_aabb[5] - obb_aabb[4]
        extents = sorted([x_extent, y_extent, z_extent])
        self.assertAlmostEqual(extents[0], 2.0, places=5)
        self.assertAlmostEqual(extents[1], 3.0, places=5)
        self.assertAlmostEqual(extents[2], 4.0, places=5)

    def test_transform_is_4x4(self) -> None:
        rng = np.random.RandomState(0)
        points = rng.randn(20, 3)
        result = _fit_gravity_aligned_obb(points)
        self.assertIsNotNone(result)
        obb_transform, obb_aabb, position = result
        self.assertEqual(obb_transform.shape, (4, 4))
        self.assertEqual(obb_aabb.shape, (6,))
        self.assertEqual(position.shape, (3,))

    def test_y_axis_preserved(self) -> None:
        """Gravity-aligned OBB should have Y-axis = [0, 1, 0] in the rotation."""
        rng = np.random.RandomState(1)
        points = rng.randn(30, 3)
        result = _fit_gravity_aligned_obb(points)
        self.assertIsNotNone(result)
        obb_transform, _, _ = result
        y_axis = obb_transform[:3, 1]
        np.testing.assert_allclose(y_axis, [0.0, 1.0, 0.0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
