from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.action import ActionBuilder
from fathom.schemas.actions import CoordinateSystem
from fathom.schemas.gemini_tools import ExecuteAction, GeminiBBox
from fathom.schemas.tools import AcceptedCommand


class ActionBuilderTest(unittest.TestCase):
    """
    Covers deterministic action materialization after command acceptance.
    """

    def test_coerces_malformed_normalized_bbox_to_logical(self) -> None:
        """
        Pixel-scale bbox values mislabeled as normalized must not survive materialization.
        """

        action = ActionBuilder().build(
            command=AcceptedCommand(
                action_type=ActionType.SWIPE_UP,
                payload=ExecuteAction(
                    action_type="swipe_up",
                    target_name="Restaurant list area",
                    scroll_target="Restaurant list area",
                    confidence=0.84,
                    bbox=GeminiBBox(
                        x=0,
                        y=858,
                        width=1206,
                        height=1396,
                        coordinate_system="normalized",
                    ),
                ),
            )
        )

        assert action.bounds is not None
        self.assertEqual(action.bounds.system, CoordinateSystem.LOGICAL)

    def test_keeps_valid_normalized_bbox_normalized(self) -> None:
        """
        Legitimate normalized bboxes keep their declared coordinate system.
        """

        action = ActionBuilder().build(
            command=AcceptedCommand(
                action_type=ActionType.SWIPE_UP,
                payload=ExecuteAction(
                    action_type="swipe_up",
                    target_name="Restaurant list area",
                    scroll_target="Restaurant list area",
                    confidence=0.84,
                    bbox=GeminiBBox(
                        x=100,
                        y=200,
                        width=800,
                        height=500,
                        coordinate_system="normalized",
                    ),
                ),
            )
        )

        assert action.bounds is not None
        self.assertEqual(action.bounds.system, CoordinateSystem.NORMALIZED)

    def test_scroll_objective_does_not_replace_surface_target(self) -> None:
        """
        scroll_target remains the objective while the executable surface uses the scroll area.
        """

        action = ActionBuilder().build(
            command=AcceptedCommand(
                action_type=ActionType.SWIPE_UP,
                payload=ExecuteAction(
                    action_type="swipe_up",
                    scroll_target="Asha Tiffin",
                    confidence=0.84,
                    bbox=GeminiBBox(
                        x=0,
                        y=858,
                        width=1206,
                        height=1396,
                        coordinate_system="pixel",
                    ),
                ),
            )
        )

        self.assertEqual(action.target, "main scrollable area")
        self.assertEqual(action.scroll_target, "Asha Tiffin")

    def test_type_text_materializes_from_payload(self) -> None:
        """
        Type commands carry text and target fields from the accepted payload.
        """

        action = ActionBuilder().build(
            command=AcceptedCommand(
                action_type=ActionType.TYPE,
                payload=ExecuteAction(
                    action_type="type",
                    target_name="Search box",
                    export_target="Search box",
                    text="soap",
                    confidence=0.91,
                ),
            )
        )

        self.assertEqual(action.action_type, ActionType.TYPE)
        self.assertEqual(action.target, "Search box")
        self.assertEqual(action.text, "soap")
