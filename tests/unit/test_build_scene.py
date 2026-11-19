# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from r3d.pipeline.frame_data import CameraIntrinsics
from r3d.pipeline.scripts import build_scene
from r3d.pipeline.scripts.build_scene import (
    _build_parser,
    _filter_outliers_knn,
    _fit_gravity_aligned_obb,
    _fused_object_points,
    _first_frame_box,
    _first_frame_object_box,
    _lift_depth_to_3d,
    _median_box,
    _median_box_from_frames,
    _median_object_box,
    _median_yaw,
    _validate_args,
    ObjectBoxConfig,
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


def _box_points(
    center: tuple[float, float, float],
    long_ext: float,
    short_ext: float,
    height: float,
    yaw: float,
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Uniform points inside a gravity-aligned box with long axis at ``yaw``."""
    rng = np.random.RandomState(seed)
    local = (rng.rand(n, 3) - 0.5) * np.array([long_ext, height, short_ext])
    c, s = np.cos(yaw), np.sin(yaw)
    x = local[:, 0] * c - local[:, 2] * s
    z = local[:, 0] * s + local[:, 2] * c
    return np.stack([x, local[:, 1], z], axis=1) + np.array(center)


def _box_dims(obb_aabb: np.ndarray) -> np.ndarray:
    return np.array(
        [
            obb_aabb[1] - obb_aabb[0],
            obb_aabb[3] - obb_aabb[2],
            obb_aabb[5] - obb_aabb[4],
        ]
    )


def _long_axis_yaw(obb_transform: np.ndarray, obb_aabb: np.ndarray) -> float:
    dims = _box_dims(obb_aabb)
    axis = obb_transform[:3, 0] if dims[0] >= dims[2] else obb_transform[:3, 2]
    return float(np.mod(np.arctan2(axis[2], axis[0]), np.pi))


def _yaw_dist(a: float, b: float) -> float:
    d = np.mod(a - b, np.pi)
    return float(min(d, np.pi - d))


class TestMedianYaw(unittest.TestCase):
    def test_wraps_around_zero(self) -> None:
        """Yaws straddling 0 == pi must give ~0, not the naive mean pi/2."""
        yaws = np.array([0.02, np.pi - 0.02, 0.04, np.pi - 0.01, 0.0])
        self.assertLess(_yaw_dist(_median_yaw(yaws), 0.0), 0.03)

    def test_plain_median_without_wrap(self) -> None:
        yaws = np.array([0.5, 0.6, 0.7, 0.8, 2.0])
        self.assertAlmostEqual(_median_yaw(yaws), 0.7, places=6)


class TestMedianBox(unittest.TestCase):
    def test_single_frame_matches_fit(self) -> None:
        pts = _box_points((1.0, 0.5, -2.0), 0.8, 0.3, 0.4, yaw=0.6)
        fit = _fit_gravity_aligned_obb(pts)
        t, a, pos = _median_box([fit])
        np.testing.assert_allclose(pos, fit[2], atol=1e-9)
        np.testing.assert_allclose(
            sorted(_box_dims(a)), sorted(_box_dims(fit[1])), atol=1e-9
        )
        self.assertLess(_yaw_dist(_long_axis_yaw(t, a), _long_axis_yaw(*fit[:2])), 1e-9)
        np.testing.assert_allclose(t[:3, 1], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.det(t[:3, :3])), 1.0, places=9)

    def test_median_rejects_outlier_frame(self) -> None:
        boxes = [
            _fit_gravity_aligned_obb(_box_points((0, 0, 0), 1.0, 0.5, 0.4, 0.3, seed=i))
            for i in range(4)
        ]
        boxes.append(
            _fit_gravity_aligned_obb(_box_points((5, 1, 5), 4.0, 3.0, 2.0, 1.4))
        )
        t, a, pos = _median_box(boxes)
        np.testing.assert_allclose(pos, [0, 0, 0], atol=0.05)
        np.testing.assert_allclose(_box_dims(a), [1.0, 0.4, 0.5], atol=0.05)
        self.assertLess(_yaw_dist(_long_axis_yaw(t, a), 0.3), 0.05)

    def test_axis_order_and_sign_ambiguity(self) -> None:
        """Same 1.0 x 0.5 box reported with flipped axis signs, swapped PCA
        axis order, and yaw straddling the 0/pi wrap."""

        def box(axis0_yaw: float, ext0: float, ext2: float, sign: float) -> tuple:
            t = np.eye(4)
            t[:3, 0] = [np.cos(axis0_yaw), 0.0, np.sin(axis0_yaw)]
            t[:3, 2] = sign * np.array([-np.sin(axis0_yaw), 0.0, np.cos(axis0_yaw)])
            aabb = np.array([-ext0, ext0, -0.2, 0.2, -ext2, ext2]) / 2.0
            return t, aabb, t[:3, 3].copy()

        long_yaw = 0.02
        boxes = [
            box(long_yaw, 1.0, 0.5, 1.0),
            box(long_yaw + np.pi, 1.0, 0.5, -1.0),
            box(long_yaw + np.pi / 2, 0.5, 1.0, 1.0),
            box(-0.03, 1.0, 0.5, 1.0),
            box(-0.03 + np.pi / 2, 0.5, 1.0, -1.0),
        ]
        t, a, _ = _median_box(boxes)
        np.testing.assert_allclose(_box_dims(a), [1.0, 0.2, 0.5], atol=1e-9)
        self.assertLess(_yaw_dist(_long_axis_yaw(t, a), long_yaw), 1e-9)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            _median_box([])


class TestMedianBoxFromFrames(unittest.TestCase):
    def test_drops_small_frames(self) -> None:
        good = [_box_points((0, 0, 0), 1.0, 0.5, 0.4, 0.3, seed=i) for i in range(3)]
        tiny = _box_points((9, 9, 9), 3.0, 3.0, 3.0, 0.0, n=10)
        box, kept = _median_box_from_frames(good + [tiny], 6, True, 50)
        self.assertIsNotNone(box)
        self.assertEqual(len(kept), 3)
        np.testing.assert_allclose(box[2], [0, 0, 0], atol=0.05)

    def test_no_knn_keeps_frame_outliers(self) -> None:
        frame = np.vstack(
            [_box_points((0, 0, 0), 1.0, 0.5, 0.4, 0.0), [[50.0, 0.0, 0.0]]]
        )
        box_knn, _ = _median_box_from_frames([frame], 6, True, 50)
        box_raw, _ = _median_box_from_frames([frame], 6, False, 50)
        self.assertLess(_box_dims(box_knn[1])[0], 1.5)
        self.assertGreater(_box_dims(box_raw[1]).max(), 40.0)

    def test_no_valid_frame_returns_none(self) -> None:
        box, kept = _median_box_from_frames([np.zeros((5, 3))], 6, True, 50)
        self.assertIsNone(box)
        self.assertEqual(kept, [])

    def test_median_object_box_fails_loudly(self) -> None:
        cfg = ObjectBoxConfig(box_mode="median", use_vote=False)
        with self.assertRaises(RuntimeError):
            _median_object_box([np.zeros((5, 3))], 7, cfg)


class TestFusedObjectPoints(unittest.TestCase):
    """Order and on/off switches of the fused recipe (KNN -> vote -> KNN)."""

    def _run(self, cfg: ObjectBoxConfig) -> list[str]:
        calls: list[str] = []

        def fake_knn(points: np.ndarray, k: int) -> np.ndarray:
            calls.append(f"knn{k}")
            return points[:-1]

        def fake_vote(points: np.ndarray, *args: object) -> np.ndarray:
            calls.append("vote")
            return points[:-2]

        pts = np.arange(30, dtype=np.float64).reshape(10, 3)
        with mock.patch.object(build_scene, "_filter_outliers_knn", fake_knn):
            with mock.patch.object(
                build_scene, "_multiview_consensus_filter", fake_vote
            ):
                out = _fused_object_points(pts, 1, "seq", None, None, cfg)
        calls.append(f"n={len(out)}")
        return calls

    def test_default_recipe(self) -> None:
        self.assertEqual(self._run(ObjectBoxConfig()), ["knn6", "vote", "knn6", "n=6"])

    def test_no_knn(self) -> None:
        self.assertEqual(self._run(ObjectBoxConfig(use_knn=False)), ["vote", "n=8"])

    def test_no_vote(self) -> None:
        self.assertEqual(
            self._run(ObjectBoxConfig(use_vote=False)), ["knn6", "knn6", "n=8"]
        )


class TestFirstFrameBox(unittest.TestCase):
    def _frames(self) -> list[tuple[int, np.ndarray]]:
        return [
            (300, _box_points((4, 0, 4), 2.0, 1.0, 1.0, 0.0, seed=3)),
            (100, _box_points((2, 0, 1), 0.6, 0.2, 0.3, 1.0, seed=2)),
            (200, _box_points((0, 0, 0), 1.0, 0.5, 0.4, 0.3, seed=1)),
        ]

    def test_uses_earliest_timestamp_only(self) -> None:
        box, pts, ts, idx = _first_frame_box(self._frames(), 6, True, 10)
        self.assertEqual((ts, idx), (100, 0))
        np.testing.assert_allclose(box[2], [2, 0, 1], atol=0.03)
        np.testing.assert_allclose(sorted(_box_dims(box[1])), [0.2, 0.3, 0.6], atol=0.03)
        self.assertLessEqual(len(pts), 2000)

    def test_matches_single_frame_fit(self) -> None:
        frame = _box_points((1, 0, 1), 0.8, 0.4, 0.5, 0.7, seed=4)
        box, _, _, _ = _first_frame_box([(5, frame)], 6, False, 10)
        np.testing.assert_allclose(box[0], _fit_gravity_aligned_obb(frame)[0])

    def test_knn_applies_to_the_frame(self) -> None:
        frame = np.vstack([_box_points((0, 0, 0), 1.0, 0.5, 0.4, 0.0), [[50.0, 0, 0]]])
        box_knn, _, _, _ = _first_frame_box([(1, frame)], 6, True, 10)
        box_raw, _, _, _ = _first_frame_box([(1, frame)], 6, False, 10)
        self.assertLess(_box_dims(box_knn[1]).max(), 1.5)
        self.assertGreater(_box_dims(box_raw[1]).max(), 40.0)

    def test_skips_small_and_empty_early_frames(self) -> None:
        frames = self._frames() + [
            (50, _box_points((9, 9, 9), 1, 1, 1, 0, n=5)),
            (60, np.zeros((0, 3))),
        ]
        box, _, ts, idx = _first_frame_box(frames, 6, True, 10)
        self.assertEqual((ts, idx), (100, 2))
        np.testing.assert_allclose(box[2], [2, 0, 1], atol=0.03)

    def test_no_valid_frame_fails(self) -> None:
        frames = [(1, np.zeros((0, 3))), (2, _box_points((0, 0, 0), 1, 1, 1, 0, n=5))]
        with self.assertRaisesRegex(RuntimeError, "none of 2 mask frames"):
            _first_frame_box(frames, 6, True, 10)

    def test_duplicate_timestamp_fails(self) -> None:
        frames = self._frames() + [(200, _box_points((0, 0, 0), 1, 1, 1, 0))]
        with self.assertRaisesRegex(RuntimeError, "share"):
            _first_frame_box(frames, 6, True, 10)

    def test_object_box_error_names_object(self) -> None:
        cfg = ObjectBoxConfig(box_mode="first_frame", use_vote=False)
        with self.assertRaisesRegex(RuntimeError, "obj 7"):
            _first_frame_object_box([(1, np.zeros((5, 3)))], 7, cfg)


class TestAblationArgs(unittest.TestCase):
    _REQ = ["--frames-dir", "/f", "--output-dir", "/o"]

    def test_defaults_match_paper_config(self) -> None:
        args = _build_parser().parse_args(self._REQ)
        self.assertFalse(args.no_knn)
        self.assertFalse(args.no_multiview_vote)
        self.assertEqual(args.box_mode, "fused")
        self.assertEqual(ObjectBoxConfig(), ObjectBoxConfig(knn_k=args.knn_k))

    def test_median_requires_no_vote(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(self._REQ + ["--box-mode", "median"])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _validate_args(parser, args)
        ok = parser.parse_args(
            self._REQ + ["--box-mode", "median", "--no-multiview-vote"]
        )
        _validate_args(parser, ok)

    def test_first_frame_requires_no_vote(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(self._REQ + ["--box-mode", "first_frame"])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            _validate_args(parser, args)
        ok = parser.parse_args(
            self._REQ + ["--box-mode", "first_frame", "--no-multiview-vote"]
        )
        _validate_args(parser, ok)


if __name__ == "__main__":
    unittest.main()
