from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from rich.panel import Panel

from fathom.adapters.telemetry.event_panels import render_event_panel
from fathom.constants.events import FathomEvent

# Events that render a Panel, paired with representative context.
_PANEL_EVENT_FIXTURES: Dict[FathomEvent, Dict[str, Any]] = {
    FathomEvent.INTENT_CLASSIFIED: {"should_decompose": True},
    FathomEvent.DECOMPOSITION_COMPLETE: {"sub_goals": ["open app", "tap login"]},
    FathomEvent.SUB_GOAL_STARTED: {"index": 0, "total": 3, "description": "open app"},
    FathomEvent.SUB_GOAL_COMPLETED: {"index": 1, "total": 3, "description": "tap login"},
    FathomEvent.HITL_REQUESTED: {"prompt": "Continue?"},
    FathomEvent.REASONING: {"step": 1, "reasoning": "Need to open app."},
    FathomEvent.PLANNED_ACTION: {
        "step": 1,
        "action_description": "Tap Login",
        "target": "#login",
        "confidence": 0.91,
    },
    FathomEvent.STEP_COMPLETED: {
        "step": 1,
        "success": True,
        "action_description": "Tap Login",
        "observation": "Login form visible",
    },
}

# Events that intentionally render nothing (state-only, or suppressed).
_NON_PANEL_EVENTS = {
    FathomEvent.STEP_AUDITED,
    FathomEvent.SCRIPT_GENERATED,
    FathomEvent.PROMPT_BUILT,
    FathomEvent.CONTEXT_CAPTURED,
    FathomEvent.LATENCY_PHASE,
    FathomEvent.LLM_CALL_COMPLETED,
    FathomEvent.HITL_RECEIVED,
    FathomEvent.WORKFLOW_PAUSED,
    FathomEvent.WORKFLOW_RESUMED,
    FathomEvent.WORKFLOW_CANCELLED,
    FathomEvent.WORKFLOW_COMPLETED,
}


class RenderEventPanelParityTest(unittest.TestCase):
    """
    Table-driven coverage over the full ``FathomEvent`` vocabulary so
    the shared renderer cannot drift away from the events the agent
    actually emits.
    """

    def test_every_event_is_classified(self) -> None:
        """Every FathomEvent must be either in the panel set or the non-panel set."""

        classified = set(_PANEL_EVENT_FIXTURES) | _NON_PANEL_EVENTS
        missing = set(FathomEvent) - classified
        self.assertEqual(
            missing,
            set(),
            f"FathomEvent(s) not classified in test fixtures: {missing}",
        )

    def test_panel_events_render_a_panel(self) -> None:
        for event, context in _PANEL_EVENT_FIXTURES.items():
            with self.subTest(event=event):
                panel = render_event_panel(
                    event_type=event,
                    message="ignored",
                    context=dict(context, type=event),
                )
                self.assertIsInstance(panel, Panel, f"{event} should render a Panel")

    def test_non_panel_events_return_none(self) -> None:
        for event in _NON_PANEL_EVENTS:
            with self.subTest(event=event):
                panel = render_event_panel(
                    event_type=event,
                    message="m",
                    context={"type": event},
                )
                self.assertIsNone(panel, f"{event} should not render a panel")


class RenderEventPanelEdgeCasesTest(unittest.TestCase):
    """
    Cover the non-table edge cases that the parity test can't express
    cleanly.
    """

    def test_decomposition_with_empty_sub_goals_returns_none(self) -> None:
        result: Optional[Panel] = render_event_panel(
            event_type=FathomEvent.DECOMPOSITION_COMPLETE,
            message="",
            context={"sub_goals": []},
        )
        self.assertIsNone(result)

    def test_error_level_without_known_event_returns_panel(self) -> None:
        result = render_event_panel(
            event_type=None,
            message="boom",
            context={},
            level="error",
        )
        self.assertIsInstance(result, Panel)

    def test_info_level_without_known_event_returns_none(self) -> None:
        result = render_event_panel(
            event_type=None,
            message="noise",
            context={},
            level="info",
        )
        self.assertIsNone(result)

    def test_step_completed_failure_produces_red_panel(self) -> None:
        """Success bit flips title/color but must still return a panel."""

        panel = render_event_panel(
            event_type=FathomEvent.STEP_COMPLETED,
            message="",
            context={
                "step": 2,
                "success": False,
                "action_description": "Tap Login",
                "observation": "Button not visible",
            },
        )
        self.assertIsInstance(panel, Panel)
