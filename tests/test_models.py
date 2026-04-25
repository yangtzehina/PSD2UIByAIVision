import unittest

from uiir.models import BBox


class BBoxTests(unittest.TestCase):
    def test_iou(self):
        left = BBox(0, 0, 100, 100)
        right = BBox(50, 50, 100, 100)
        self.assertAlmostEqual(left.iou(right), 2500 / 17500)

    def test_contains_bbox_with_padding(self):
        outer = BBox(10, 10, 100, 100)
        inner = BBox(8, 8, 20, 20)
        self.assertFalse(outer.contains_bbox(inner))
        self.assertTrue(outer.contains_bbox(inner, padding=3))


if __name__ == "__main__":
    unittest.main()
