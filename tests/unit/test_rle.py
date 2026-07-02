# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

import numpy as np
from r3d.utils.rle import decode_rle_to_mask, encode_mask_to_rle


class TestDecodeRleToMask(unittest.TestCase):
    def test_all_false(self) -> None:
        rle = {"counts": [6], "size": [2, 3]}
        mask = decode_rle_to_mask(rle)
        self.assertEqual(mask.shape, (2, 3))
        self.assertFalse(mask.any())

    def test_all_true(self) -> None:
        rle = {"counts": [0, 6], "size": [2, 3]}
        mask = decode_rle_to_mask(rle)
        self.assertEqual(mask.shape, (2, 3))
        self.assertTrue(mask.all())

    def test_known_pattern(self) -> None:
        # 4x4 mask, column-major (Fortran order)
        # counts: 2 false, 3 true, 5 false, 4 true, 2 false = 16
        rle = {"counts": [2, 3, 5, 4, 2], "size": [4, 4]}
        mask = decode_rle_to_mask(rle)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, bool)
        # Verify total true count
        self.assertEqual(mask.sum(), 7)

    def test_single_pixel_true(self) -> None:
        # 1x1 mask with single True
        rle = {"counts": [0, 1], "size": [1, 1]}
        mask = decode_rle_to_mask(rle)
        self.assertTrue(mask[0, 0])

    def test_raises_on_wrong_total(self) -> None:
        rle = {"counts": [5], "size": [2, 3]}  # 5 != 2*3=6
        with self.assertRaises(ValueError):
            decode_rle_to_mask(rle)

    def test_raises_on_missing_key(self) -> None:
        with self.assertRaises(KeyError):
            decode_rle_to_mask({"counts": [4]})
        with self.assertRaises(KeyError):
            decode_rle_to_mask({"size": [2, 2]})


class TestEncodeDecodeRoundtrip(unittest.TestCase):
    def test_roundtrip_random(self) -> None:
        rng = np.random.RandomState(42)
        mask = rng.randint(0, 2, size=(10, 15)).astype(bool)
        rle = encode_mask_to_rle(mask)
        decoded = decode_rle_to_mask(rle)
        np.testing.assert_array_equal(mask, decoded)

    def test_roundtrip_empty(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        rle = encode_mask_to_rle(mask)
        decoded = decode_rle_to_mask(rle)
        np.testing.assert_array_equal(mask, decoded)

    def test_roundtrip_full(self) -> None:
        mask = np.ones((3, 7), dtype=bool)
        rle = encode_mask_to_rle(mask)
        decoded = decode_rle_to_mask(rle)
        np.testing.assert_array_equal(mask, decoded)


if __name__ == "__main__":
    unittest.main()
