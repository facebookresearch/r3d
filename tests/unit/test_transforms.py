# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import math
import unittest

import numpy as np
from r3d.utils.transforms import invert_rigid_transform, rotation_matrix_to_quaternion


class TestInvertRigidTransform(unittest.TestCase):
    def test_identity(self) -> None:
        T = np.eye(4)
        T_inv = invert_rigid_transform(T)
        np.testing.assert_allclose(T_inv, np.eye(4), atol=1e-12)

    def test_pure_translation(self) -> None:
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        T_inv = invert_rigid_transform(T)
        expected = np.eye(4)
        expected[:3, 3] = [-1.0, -2.0, -3.0]
        np.testing.assert_allclose(T_inv, expected, atol=1e-12)

    def test_rotation_90_z(self) -> None:
        T = np.eye(4)
        c, s = 0.0, 1.0
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T[:3, 3] = [1.0, 0.0, 0.0]
        T_inv = invert_rigid_transform(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-12)

    def test_inverse_of_inverse_is_original(self) -> None:
        rng = np.random.RandomState(42)
        R, _ = np.linalg.qr(rng.randn(3, 3))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = rng.randn(3)
        T_back = invert_rigid_transform(invert_rigid_transform(T))
        np.testing.assert_allclose(T_back, T, atol=1e-12)

    def test_product_is_identity(self) -> None:
        angle = math.pi / 6
        T = np.eye(4)
        T[:3, :3] = [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
        T[:3, 3] = [5.0, -3.0, 2.0]
        T_inv = invert_rigid_transform(T)
        np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12)


class TestRotationMatrixToQuaternion(unittest.TestCase):
    def test_identity(self) -> None:
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_90_deg_about_z(self) -> None:
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        q = rotation_matrix_to_quaternion(R)
        expected = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        np.testing.assert_allclose(q, expected, atol=1e-12)

    def test_180_deg_about_x(self) -> None:
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        q = rotation_matrix_to_quaternion(R)
        self.assertAlmostEqual(abs(q[0]), 0.0, places=10)
        self.assertAlmostEqual(abs(q[1]), 1.0, places=10)
        self.assertAlmostEqual(abs(q[2]), 0.0, places=10)
        self.assertAlmostEqual(abs(q[3]), 0.0, places=10)

    def test_accepts_list_input(self) -> None:
        R_list = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        q = rotation_matrix_to_quaternion(R_list)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_unit_quaternion(self) -> None:
        rng = np.random.RandomState(99)
        R, _ = np.linalg.qr(rng.randn(3, 3))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        q = rotation_matrix_to_quaternion(R)
        norm = math.sqrt(sum(x * x for x in q))
        self.assertAlmostEqual(norm, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
