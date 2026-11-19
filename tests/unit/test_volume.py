# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

import numpy as np
from r3d.pipeline.volume import bbox_volume, format_volume_error, format_volume_result


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


class TestVolumeToolWording(unittest.TestCase):
    """The get_object_volume strings, shared with the internal R3D pipeline."""

    def test_mesh_rescaled(self) -> None:
        self.assertEqual(
            format_volume_result(0.0012345, "mesh_rescaled"),
            "estimated_volume=0.001234 cubic meters (1.234 liters), "
            "computed from the object's rescaled 3D mesh",
        )
        self.assertEqual(
            format_volume_error("mesh_rescaled", 12),
            "Error: could not compute rescaled mesh volume for object 12.",
        )

    def test_bbox(self) -> None:
        self.assertEqual(
            format_volume_result(0.5, "bbox"),
            "estimated_volume=0.500000 cubic meters (500.000 liters), "
            "computed from the object's 3D reconstruction",
        )
        self.assertEqual(
            format_volume_error("bbox", 3), "Error: object has no reconstruction."
        )

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(KeyError):
            format_volume_result(1.0, "gaussian")


if __name__ == "__main__":
    unittest.main()
