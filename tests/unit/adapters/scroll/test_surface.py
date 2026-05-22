import unittest

from fathom.adapters.scroll.surface import ScrollSurfaceInspector
from fathom.constants.scroll import SurfaceKind
from fathom.schemas.actions import Bounds, CoordinateSystem
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.screens import ScreenHashBundle


class ScrollSurfaceInspectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers explicit footer detection from bottom navigation elements.
    """

    async def test_detects_bottom_navigation_cluster_as_footer_hint(self) -> None:
        """
        Aggregate clustered bottom tappables into one footer hint.
        """

        observation = ScreenObservation(
            activity="com.aranoah.healthkart.plus",
            hashes=ScreenHashBundle(visual_hash="a", xml_hash="b", interaction_hash="c"),
            elements=self.__bottom_navigation(),
            overlays=tuple(),
            keyboard=KeyboardObservation(visible=False, bounds=None, dismiss=tuple()),
            scroll=tuple(),
            calls_to_action=tuple(),
            focused=None,
        )

        hints = await ScrollSurfaceInspector().inspect(
            observation=observation,
            path=self.__path(),
            capture_width=1080,
            capture_height=2340,
        )

        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].kind, SurfaceKind.FOOTER)
        self.assertEqual(hints[0].bounds.y, 1881)

    @staticmethod
    def __bottom_navigation() -> tuple[PerceivedElement, ...]:
        """
        Build one bottom navigation cluster.
        """

        return (
            PerceivedElement(
                identifier="food",
                bounds=Bounds(
                    x=87,
                    y=1881,
                    width=66,
                    height=69,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                ),
                source=ElementSource.XML,
                role=ElementRole.BUTTON,
                confidence=1.0,
                text="Food",
                tappable=True,
            ),
            PerceivedElement(
                identifier="delivery",
                bounds=Bounds(
                    x=327,
                    y=1881,
                    width=69,
                    height=69,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                ),
                source=ElementSource.XML,
                role=ElementRole.BUTTON,
                confidence=1.0,
                text="10mins Delivery",
                tappable=True,
            ),
            PerceivedElement(
                identifier="store",
                bounds=Bounds(
                    x=570,
                    y=1881,
                    width=69,
                    height=69,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                ),
                source=ElementSource.XML,
                role=ElementRole.BUTTON,
                confidence=1.0,
                text="99 store",
                tappable=True,
            ),
        )

    @staticmethod
    def __path():
        """
        Return a representative failing swipe path.
        """

        from fathom.schemas.actions import GesturePath

        return GesturePath(start_x=540, start_y=1921, end_x=540, end_y=521, duration=350)
