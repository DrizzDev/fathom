import unittest

from fathom.schemas.actions import BoundingBox
from fathom.schemas.configuration import GeminiConfig
from fathom.tools.vision.gemini import GeminiVisionTool


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
        tool = GeminiVisionTool(GeminiConfig(api_key="test", model="test"))

        # Test x, y, width, height dict
        res1 = tool._parse_bbox({"x": 100, "y": 200, "width": 50, "height": 80})
        self.assertEqual(res1, {"x": 100, "y": 200, "width": 50, "height": 80})

        # Test ymin, xmin, ymax, xmax dict
        res2 = tool._parse_bbox({"ymin": 200, "xmin": 100, "ymax": 280, "xmax": 150})
        self.assertEqual(res2, {"x": 100, "y": 200, "width": 50, "height": 80})

        # Test Gemini native array [ymin, xmin, ymax, xmax]
        res3 = tool._parse_bbox([200, 100, 280, 150])
        self.assertEqual(res3, {"x": 100, "y": 200, "width": 50, "height": 80})

        # Test invalid
        self.assertIsNone(tool._parse_bbox([1, 2, 3]))
        self.assertIsNone(tool._parse_bbox({"incorrect": "format"}))


if __name__ == "__main__":
    unittest.main()
