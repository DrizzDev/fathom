"""
Unit tests for history command description normalization.
"""

from __future__ import annotations

from fathom.services.history import HistoryService


class TestHistoryService:
    """
    Command descriptions use canonical grammar and normalized spacing.
    """

    def test_describe_command_normalizes_target_spacing(self) -> None:
        service = HistoryService(workflow_id="coherence")
        command = service._HistoryService__describe_command(
            {"action_type": "tap", "event_type": "action", "target": "  search   bar  "}
        )
        assert command == "Tap on search bar"

    def test_validation_command_uses_canonical_grammar(self) -> None:
        service = HistoryService(workflow_id="coherence")
        command = service._HistoryService__describe_command(
            {"action_type": "validate", "event_type": "validation", "target": "  profile   icon  "}
        )
        assert command == "Validate profile icon"
