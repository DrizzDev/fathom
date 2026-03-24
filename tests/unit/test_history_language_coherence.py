from fathom.services.history import HistoryService


def test_history_describe_command_normalizes_target_spacing() -> None:
    service = HistoryService(workflow_id="coherence")
    command = service._HistoryService__describe_command(
        {
            "action_type": "tap",
            "event_type": "action",
            "target": "  search   bar  ",
        }
    )
    assert command == "Tap on search bar"


def test_history_validation_command_uses_canonical_grammar() -> None:
    service = HistoryService(workflow_id="coherence")
    command = service._HistoryService__describe_command(
        {
            "action_type": "validate",
            "event_type": "validation",
            "target": "  profile   icon  ",
        }
    )
    assert command == "Validate profile icon"
