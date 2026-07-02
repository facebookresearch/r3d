# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

import numpy as np
from r3d.pipeline.volume import bbox_volume


class TestBboxVolume(unittest.TestCase):
    def test_unit_cube(self) -> None:
        # [xmin, xmax, ymin, ymax, zmin, zmax]
        aabb = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(bbox_volume(aabb), 1.0)

    def test_known_box(self) -> None:
        aabb = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
        self.assertAlmostEqual(bbox_volume(aabb), 6.0)

    def test_zero_volume(self) -> None:
        aabb = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 3.0])
        self.assertAlmostEqual(bbox_volume(aabb), 0.0)

    def test_negative_coords(self) -> None:
        aabb = np.array([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0])
        self.assertAlmostEqual(bbox_volume(aabb), 2.0 * 4.0 * 6.0)


if __name__ == "__main__":
    unittest.main()
