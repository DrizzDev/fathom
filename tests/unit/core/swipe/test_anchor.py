from __future__ import annotations

import unittest

from fathom.core.swipe.anchor import AnchorGuard
from fathom.schemas.actions import Bounds, CoordinateSystem, GesturePath
from fathom.schemas.swipe import ReservePolicy


class AnchorGuardAddressableTest(unittest.TestCase):
    """
    Verify the addressable rectangle derived from a viewport and its edge reserve.
    """

    @staticmethod
    def __viewport(width: int = 1280, height: int = 2856) -> Bounds:
        """
        Build a device-pixel viewport rectangle.
        """

        return Bounds(
            x=0,
            y=0,
            width=width,
            height=height,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def test_reserve_is_withheld_from_every_edge(self) -> None:
        """
        Each edge withholds its configured fraction of the viewport.
        """

        area = AnchorGuard().addressable(viewport=self.__viewport(), policy=ReservePolicy())

        self.assertIsNotNone(area)
        self.assertEqual(area.y, int(2856 * 0.05))
        self.assertEqual(area.x, int(1280 * 0.12))
        self.assertEqual(area.height, 2856 - int(2856 * 0.05) - int(2856 * 0.10))
        self.assertEqual(area.width, 1280 - (int(1280 * 0.12) * 2))

    def test_default_reserve_clears_the_android_gesture_inset(self) -> None:
        """
        The default bottom reserve exceeds a 48dp system gesture inset across realistic device geometries.
        """

        for height, scale in ((2856, 3.0), (2400, 3.0), (1920, 3.0), (3120, 3.5), (1600, 2.0)):
            with self.subTest(height=height, scale=scale):
                area = AnchorGuard().addressable(
                    viewport=self.__viewport(height=height),
                    policy=ReservePolicy(),
                )

                inset = int(48 * scale)
                self.assertLess(area.y + area.height, height - inset)

    def test_zero_reserve_leaves_the_viewport_whole(self) -> None:
        """
        A fully permissive policy yields the viewport unchanged.
        """

        area = AnchorGuard().addressable(
            viewport=self.__viewport(),
            policy=ReservePolicy(top=0.0, bottom=0.0, side=0.0),
        )

        self.assertEqual((area.x, area.y, area.width, area.height), (0, 0, 1280, 2856))

    def test_degenerate_viewport_has_no_addressable_area(self) -> None:
        """
        A zero-sized viewport from a failed capture yields None rather than a degenerate rectangle.
        """

        area = AnchorGuard().addressable(
            viewport=Bounds(x=0, y=0, width=0, height=0),
            policy=ReservePolicy(),
        )

        self.assertIsNone(area)

    def test_any_valid_reserve_leaves_a_usable_area(self) -> None:
        """
        The reserve ceiling guarantees a non-degenerate area for any real viewport.
        """

        policy = ReservePolicy(top=0.49, bottom=0.49, side=0.49)

        area = AnchorGuard().addressable(viewport=self.__viewport(), policy=policy)

        self.assertIsNotNone(area)
        self.assertGreaterEqual(area.width, 1)
        self.assertGreaterEqual(area.height, 1)


class AnchorGuardConfineTest(unittest.TestCase):
    """
    Verify touch-down confinement while gesture endpoints and travel are preserved.
    """

    @staticmethod
    def __viewport() -> Bounds:
        """
        Build the 1280x2856 device-pixel viewport observed in workflow 22a32299.
        """

        return Bounds(
            x=0,
            y=0,
            width=1280,
            height=2856,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def test_bottom_edge_anchor_is_pulled_clear_of_the_gesture_inset(self) -> None:
        """
        The workflow 22a32299 swipe anchored 64px above the screen bottom is pulled above the 48dp inset.
        """

        original = GesturePath(start_x=640, start_y=2792, end_x=640, end_y=64, duration=300)

        confined = AnchorGuard().confine(
            path=original,
            viewport=self.__viewport(),
            policy=ReservePolicy(),
        )

        self.assertIsNotNone(confined)
        self.assertLess(confined.start_y, 2856 - int(48 * 3.0))
        self.assertEqual(confined.end_y, original.end_y)
        self.assertEqual(confined.start_x, original.start_x)

    def test_right_edge_anchor_is_pulled_clear_of_the_back_gesture_inset(self) -> None:
        """
        A leftward swipe anchored at the right edge is pulled clear of a 40dp back-gesture inset.
        """

        original = GesturePath(start_x=1216, start_y=1428, end_x=64, end_y=1428, duration=300)

        confined = AnchorGuard().confine(
            path=original,
            viewport=self.__viewport(),
            policy=ReservePolicy(),
        )

        self.assertLess(confined.start_x, 1280 - int(40 * 3.0))
        self.assertEqual(confined.end_x, original.end_x)

    def test_left_edge_anchor_is_pulled_clear_of_the_back_gesture_inset(self) -> None:
        """
        A rightward swipe anchored at the left edge is pulled clear of a 40dp back-gesture inset.
        """

        original = GesturePath(start_x=64, start_y=1428, end_x=1216, end_y=1428, duration=300)

        confined = AnchorGuard().confine(
            path=original,
            viewport=self.__viewport(),
            policy=ReservePolicy(),
        )

        self.assertGreater(confined.start_x, int(40 * 3.0))
        self.assertEqual(confined.end_x, original.end_x)

    def test_anchor_already_addressable_is_returned_unchanged(self) -> None:
        """
        A gesture anchored inside the addressable area is not modified.
        """

        original = GesturePath(start_x=640, start_y=1352, end_x=640, end_y=848, duration=300)

        confined = AnchorGuard().confine(
            path=original,
            viewport=self.__viewport(),
            policy=ReservePolicy(),
        )

        self.assertIs(confined, original)

    def test_confinement_preserves_the_endpoint(self) -> None:
        """
        Only the touch-down moves; the endpoint may remain inside a reserved edge.
        """

        original = GesturePath(start_x=640, start_y=2792, end_x=640, end_y=2800, duration=300)

        confined = AnchorGuard().confine(
            path=original,
            viewport=self.__viewport(),
            policy=ReservePolicy(),
        )

        self.assertEqual(confined.end_y, 2800)
        self.assertNotEqual(confined.start_y, original.start_y)

    def test_no_addressable_area_yields_no_path(self) -> None:
        """
        Confinement reports None when the reserve leaves no addressable area to anchor in.
        """

        confined = AnchorGuard().confine(
            path=GesturePath(start_x=2, start_y=2, end_x=3, end_y=3, duration=300),
            viewport=Bounds(x=0, y=0, width=0, height=0),
            policy=ReservePolicy(),
        )

        self.assertIsNone(confined)
