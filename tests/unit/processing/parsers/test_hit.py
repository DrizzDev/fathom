from __future__ import annotations

import unittest

from fathom.constants.screen import HitOutcome
from fathom.processing.parsers.hit import InteractiveHitTester

_BUTTON = (
    "<hierarchy>"
    '<node class="android.widget.Button" clickable="true" enabled="true" '
    'bounds="[100,200][300,260]"/>'
    '<node class="android.widget.TextView" clickable="false" enabled="true" '
    'bounds="[0,0][500,50]"/>'
    "</hierarchy>"
)


class InteractiveHitTesterTest(unittest.TestCase):
    """
    Verifies hit/miss/unknown classification of a tap over a hierarchy.
    """

    def test_point_inside_an_interactive_element_is_a_hit(self) -> None:
        outcome = InteractiveHitTester.locate(xml_content=_BUTTON, point_x=200, point_y=230)
        self.assertEqual(outcome, HitOutcome.HIT)

    def test_point_off_every_interactive_element_is_a_miss(self) -> None:
        outcome = InteractiveHitTester.locate(xml_content=_BUTTON, point_x=400, point_y=400)
        self.assertEqual(outcome, HitOutcome.MISS)

    def test_a_tree_without_interactive_elements_is_unknown(self) -> None:
        xml = (
            "<hierarchy>"
            '<node class="android.widget.TextView" clickable="false" bounds="[0,0][500,50]"/>'
            "</hierarchy>"
        )
        outcome = InteractiveHitTester.locate(xml_content=xml, point_x=10, point_y=10)
        self.assertEqual(outcome, HitOutcome.UNKNOWN)

    def test_disabled_interactive_element_does_not_count_as_a_hit(self) -> None:
        xml = (
            "<hierarchy>"
            '<node class="android.widget.Button" clickable="true" enabled="false" '
            'bounds="[0,0][100,100]"/>'
            '<node class="android.widget.Button" clickable="true" enabled="true" '
            'bounds="[200,200][300,300]"/>'
            "</hierarchy>"
        )
        outcome = InteractiveHitTester.locate(xml_content=xml, point_x=50, point_y=50)
        self.assertEqual(outcome, HitOutcome.MISS)

    def test_absent_hierarchy_is_unknown(self) -> None:
        self.assertEqual(
            InteractiveHitTester.locate(xml_content=None, point_x=0, point_y=0),
            HitOutcome.UNKNOWN,
        )

    def test_unparseable_hierarchy_is_unknown(self) -> None:
        self.assertEqual(
            InteractiveHitTester.locate(xml_content="<not-xml", point_x=0, point_y=0),
            HitOutcome.UNKNOWN,
        )


if __name__ == "__main__":
    unittest.main()
