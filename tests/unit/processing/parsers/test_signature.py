from __future__ import annotations

import unittest

from fathom.constants.screen import ZERO_HASH
from fathom.processing.parsers.signature import HierarchySignatureBuilder


def _hierarchy(*, nodes: str) -> str:
    return f'<hierarchy package="com.app">{nodes}</hierarchy>'


_EDIT = (
    '<node class="android.widget.EditText" resource-id="com.app:id/phone" '
    'bounds="[0,0][100,50]" clickable="true" text="{text}" />'
)
_CARD = (
    '<node class="android.widget.TextView" resource-id="com.app:id/card" '
    'bounds="[0,{top}][100,{bottom}]" text="{text}" />'
)


class HierarchyLayoutHashTest(unittest.TestCase):
    """The layout hash captures structure (class + id) and ignores text and count."""

    def setUp(self) -> None:
        self.__builder = HierarchySignatureBuilder()

    def test_text_does_not_change_the_layout_hash(self) -> None:
        empty = self.__builder.compute_layout_hash(
            xml_content=_hierarchy(nodes=_EDIT.format(text=""))
        )
        filled = self.__builder.compute_layout_hash(
            xml_content=_hierarchy(nodes=_EDIT.format(text="9876543210"))
        )

        self.assertEqual(empty, filled)
        self.assertNotEqual(empty, ZERO_HASH)

    def test_item_count_does_not_change_the_layout_hash(self) -> None:
        one = self.__builder.compute_layout_hash(
            xml_content=_hierarchy(nodes=_CARD.format(top=0, bottom=50, text="A"))
        )
        many = self.__builder.compute_layout_hash(
            xml_content=_hierarchy(
                nodes=(
                    _CARD.format(top=0, bottom=50, text="A")
                    + _CARD.format(top=60, bottom=110, text="B")
                    + _CARD.format(top=120, bottom=170, text="C")
                )
            )
        )

        self.assertEqual(one, many)

    def test_different_structure_changes_the_layout_hash(self) -> None:
        editable = self.__builder.compute_layout_hash(
            xml_content=_hierarchy(nodes=_EDIT.format(text="x"))
        )
        listing = self.__builder.compute_layout_hash(
            xml_content=_hierarchy(nodes=_CARD.format(top=0, bottom=50, text="x"))
        )

        self.assertNotEqual(editable, listing)


if __name__ == "__main__":
    unittest.main()
