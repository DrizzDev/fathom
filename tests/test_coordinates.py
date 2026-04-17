import unittest
from unittest.mock import MagicMock

from fathom.schemas.actions import BoundingBox, Bounds
from fathom.tools.vision.gemini import GeminiVisionTool
from fathom.utils.coordinates import CoordinateConverter


class TestCoordinates(unittest.TestCase):
    def test_bounding_box_center(self):
        bbox = BoundingBox(x=100, y=200, width=50, height=80)
        self.assertEqual(bbox.center_x, 125)
        self.assertEqual(bbox.center_y, 240)

    def test_bounding_box_to_pixels(self):
        bbox = BoundingBox(x=500, y=500, width=200, height=200)
        # Normalized 500 on 1000 scale is 0.5
        # 1080 * 0.5 = 540
        # 1920 * 0.5 = 960
        px_x, px_y, px_w, px_h = bbox.to_pixels(1080, 1920)
        self.assertEqual(px_x, 540)
        self.assertEqual(px_y, 960)
        self.assertEqual(px_w, 216)
        self.assertEqual(px_h, 384)

    def test_parse_bbox_formats(self):
        tool = GeminiVisionTool(
            model=MagicMock(),
            memory=MagicMock(),
            ledger=MagicMock(),
            local_storage=MagicMock(),
        )

        # Test x, y, width, height dict
        res1 = tool._GeminiVisionTool__parse_bbox({"x": 100, "y": 200, "width": 50, "height": 80})
        self.assertEqual(res1, {"x": 100, "y": 200, "width": 50, "height": 80})

        # Test ymin, xmin, ymax, xmax dict
        res2 = tool._GeminiVisionTool__parse_bbox(
            {"ymin": 200, "xmin": 100, "ymax": 280, "xmax": 150}
        )
        self.assertEqual(res2, {"x": 100, "y": 200, "width": 50, "height": 80})

        # Test Gemini native array [ymin, xmin, ymax, xmax]
        res3 = tool._GeminiVisionTool__parse_bbox([200, 100, 280, 150])
        self.assertEqual(res3, {"x": 100, "y": 200, "width": 50, "height": 80})

        # Test invalid
        self.assertIsNone(tool._GeminiVisionTool__parse_bbox([1, 2, 3]))
        self.assertIsNone(tool._GeminiVisionTool__parse_bbox({"incorrect": "format"}))


class TestSwipeCoordinates(unittest.TestCase):
    """swipe_coordinates pivots on the LLM-decided coordinates and produces
    a fixed-magnitude swipe regardless of bounds size or screen size."""

    def setUp(self):
        # 1080 x 1920 portrait-phone screen
        self.converter = CoordinateConverter(screen_width=1080, screen_height=1920)

    def test_point_bounds_pivot_and_fixed_vertical_distance(self):
        # LLM tap_target at (500, 500) normalized → pixel center (540, 960).
        point = Bounds(x=500, y=500, width=0, height=0)
        x1, y1, x2, y2 = self.converter.swipe_coordinates(bounds=point, direction="up")

        # Pivot preserved on the horizontal axis.
        self.assertEqual(x1, 540)
        self.assertEqual(x2, 540)
        # Fixed magnitude: 350px total, half (175) on each side of pivot y=960.
        self.assertEqual(y1, 960 + 175)
        self.assertEqual(y2, 960 - 175)

    def test_point_bounds_fixed_horizontal_distance(self):
        point = Bounds(x=500, y=500, width=0, height=0)
        x1, y1, x2, y2 = self.converter.swipe_coordinates(bounds=point, direction="left")
        self.assertEqual(y1, 960)
        self.assertEqual(y2, 960)
        self.assertEqual(abs(x1 - x2), 350)

    def test_box_bounds_still_produce_fixed_distance(self):
        # Non-zero bounds no longer produce bounds-proportional distance —
        # magnitude is the same constant as for point bounds.
        box = Bounds(x=200, y=200, width=600, height=600)
        _, y1, _, y2 = self.converter.swipe_coordinates(bounds=box, direction="up")
        self.assertEqual(abs(y1 - y2), 350)

    def test_tiny_bounds_still_produce_fixed_distance(self):
        # Random-exploration's Bounds(x, y, 50, 50): old math gave 35px, new
        # math gives the fixed 350px regardless.
        tiny = Bounds(x=500, y=500, width=50, height=50)
        x1, _, x2, _ = self.converter.swipe_coordinates(bounds=tiny, direction="right")
        self.assertEqual(abs(x1 - x2), 350)

    def test_swipe_endpoints_clamped_to_viewport_near_edges(self):
        # Pivot near the bottom-right — swipe-down must not exit screen bounds,
        # even though the full fixed magnitude wouldn't fit on that side.
        near_edge = Bounds(x=950, y=950, width=0, height=0)
        _, y1, _, y2 = self.converter.swipe_coordinates(bounds=near_edge, direction="down")
        self.assertLess(max(y1, y2), 1920)

    def test_scroll_from_llm_point_uses_fixed_distance(self):
        # Scroll actions from the LLM flow through execution.py:91-102 →
        # swipe_coordinates(direction="up"). Assert the resulting scroll is
        # the fixed magnitude, not a device-dependent fraction.
        point = Bounds(x=500, y=500, width=0, height=0)
        _, y1, _, y2 = self.converter.swipe_coordinates(bounds=point, direction="up")
        self.assertEqual(abs(y1 - y2), 350)


if __name__ == "__main__":
    unittest.main()
