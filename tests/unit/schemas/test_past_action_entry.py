from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.schemas.vision import (
    ActionKind,
    PastActionEntry,
    action_kind_for,
    action_kind_from_token,
)


class ActionKindDerivationTest(unittest.TestCase):
    """
    Pins the :class:`ActionType` -> :class:`ActionKind` mapping.
    """

    def test_tap_swipe_scroll_navigation(self) -> None:
        """
        Spatial gestures classify as NAVIGATION with expected screen change.
        """

        for action_type in (
            ActionType.TAP,
            ActionType.SWIPE,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
            ActionType.SCROLL,
            ActionType.LONG_PRESS,
            ActionType.BACK,
            ActionType.HOME,
            ActionType.HIDE_KEYBOARD,
        ):
            with self.subTest(action_type=action_type):
                self.assertIs(action_kind_for(action_type), ActionKind.NAVIGATION)

    def test_type_action_is_input_not_navigation(self) -> None:
        """
        TYPE is distinctly INPUT, not NAVIGATION (fix for the overlapping-kind bug).
        """

        self.assertIs(action_kind_for(ActionType.TYPE), ActionKind.INPUT)

    def test_validate_is_validation(self) -> None:
        """
        VALIDATE classifies as VALIDATION.
        """

        self.assertIs(action_kind_for(ActionType.VALIDATE), ActionKind.VALIDATION)

    def test_ask_user_is_escalation(self) -> None:
        """
        ASK_USER classifies as ESCALATION so the LLM can recognise prior HITL.
        """

        self.assertIs(action_kind_for(ActionType.ASK_USER), ActionKind.ESCALATION)

    def test_complete_is_terminal(self) -> None:
        """
        COMPLETE classifies as TERMINAL.
        """

        self.assertIs(action_kind_for(ActionType.COMPLETE), ActionKind.TERMINAL)

    def test_wait_is_observation(self) -> None:
        """
        WAIT classifies as OBSERVATION.
        """

        self.assertIs(action_kind_for(ActionType.WAIT), ActionKind.OBSERVATION)

    def test_token_lookup_tolerates_unknown(self) -> None:
        """
        Unknown raw tokens map to :attr:`ActionKind.UNKNOWN` rather than raise.
        """

        self.assertIs(action_kind_from_token("not_a_real_action"), ActionKind.UNKNOWN)

    def test_token_lookup_is_case_tolerant(self) -> None:
        """
        Tokens are lower-cased before lookup so casing differences are tolerated.
        """

        self.assertIs(action_kind_from_token("TAP"), ActionKind.NAVIGATION)

    def test_swipe_and_scroll_tokens_resolve_to_navigation(self) -> None:
        """
        Swipe-family and scroll tokens must resolve to NAVIGATION so loop evidence stays classified as active.
        """

        for token in ("swipe_left", "swipe_right", "swipe_up", "swipe_down", "scroll"):
            with self.subTest(token=token):
                self.assertIs(action_kind_from_token(token), ActionKind.NAVIGATION)


class PastActionEntryTest(unittest.TestCase):
    """
    Pins :class:`PastActionEntry` construction from raw history dicts.
    """

    def test_validate_entry_expects_no_screen_change(self) -> None:
        """
        VALIDATION entries flag ``expected_screen_change=False`` so the LLM
        does not mis-interpret no-progress as evidence of being stuck.
        """

        entry = PastActionEntry.from_raw(entry={"action": "validate", "target": "srp"})
        self.assertIs(entry.kind, ActionKind.VALIDATION)
        self.assertFalse(entry.expected_screen_change)
        self.assertEqual(entry.action, "validate")
        self.assertEqual(entry.target, "srp")

    def test_tap_entry_expects_screen_change(self) -> None:
        """
        NAVIGATION entries flag ``expected_screen_change=True``.
        """

        entry = PastActionEntry.from_raw(entry={"action": "tap", "target": "Search"})
        self.assertIs(entry.kind, ActionKind.NAVIGATION)
        self.assertTrue(entry.expected_screen_change)

    def test_type_entry_expects_screen_change(self) -> None:
        """
        INPUT entries flag ``expected_screen_change=True``.
        """

        entry = PastActionEntry.from_raw(entry={"action": "type", "target": "search box"})
        self.assertIs(entry.kind, ActionKind.INPUT)
        self.assertTrue(entry.expected_screen_change)

    def test_ask_user_entry_annotated_as_escalation(self) -> None:
        """
        Prior ASK_USER entries are annotated so the LLM does not imitate HITL.
        """

        entry = PastActionEntry.from_raw(entry={"action": "ask_user", "target": ""})
        self.assertIs(entry.kind, ActionKind.ESCALATION)
        self.assertFalse(entry.expected_screen_change)

    def test_missing_action_token_maps_to_unknown(self) -> None:
        """
        An entry without an action token defaults to UNKNOWN, not a crash.
        """

        entry = PastActionEntry.from_raw(entry={})
        self.assertEqual(entry.action, "unknown")
        self.assertIs(entry.kind, ActionKind.UNKNOWN)
        self.assertFalse(entry.expected_screen_change)

    def test_sub_goal_index_extracted(self) -> None:
        """
        ``sub_goal_index`` is surfaced when present in the raw dict.
        """

        entry = PastActionEntry.from_raw(
            entry={"action": "validate", "target": "srp", "sub_goal_index": 6}
        )
        self.assertEqual(entry.sub_goal_index, 6)

    def test_sub_goal_index_missing_stays_none(self) -> None:
        """
        ``sub_goal_index`` is optional and stays None when absent.
        """

        entry = PastActionEntry.from_raw(entry={"action": "tap", "target": "x"})
        self.assertIsNone(entry.sub_goal_index)

    def test_kind_derived_from_action_token_authoritatively(self) -> None:
        """
        ``kind`` is derived from the action token, ignoring any conflicting metadata.
        """

        entry = PastActionEntry.from_raw(
            entry={
                "action": "validate",
                "target": "x",
                "event_type": "navigation",  # conflicting stale metadata
            }
        )
        self.assertIs(entry.kind, ActionKind.VALIDATION)
