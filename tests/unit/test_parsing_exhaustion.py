from unittest.mock import MagicMock

from fathom.services.parsing import ToolResponseParser


class MockPart:
    def __init__(self, function_call=None, text=None):
        self.function_call = function_call
        self.text = text


class MockContent:
    def __init__(self, parts):
        self.parts = parts


class MockCandidate:
    def __init__(self, content):
        self.content = content
        self.finish_reason = "STOP"


class MockResponse:
    def __init__(self, candidates):
        self.candidates = candidates


def test_parse_content_exhausted_flag():
    parser = ToolResponseParser()

    # Mock tool call with content_exhausted=True
    function_call = MagicMock()
    function_call.name = "execute_ui"
    function_call.args = {
        "action": {"action_type": "swipe_left", "rationale": "next item", "is_valid": True},
        "assistant_message": "Swiping left",
        "content_exhausted": True,
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.content_exhausted is True


def test_parse_content_exhausted_default_false():
    parser = ToolResponseParser()

    # Mock tool call without flag
    function_call = MagicMock()
    function_call.name = "execute_ui"
    function_call.args = {
        "action": {"action_type": "swipe_left", "rationale": "next item", "is_valid": True},
        "assistant_message": "Swiping left",
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.content_exhausted is False


def test_parse_validate_state_sets_validation_event_type():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "validate_state"
    function_call.args = {
        "assistant_message": "Validated price is shown",
        "evidence": "Price label visible on checkout screen",
        "goal_completed": False,
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.metadata.get("event_type") == "validation"
    assert result.metadata.get("tool_name") == "validate_state"


def test_parse_verify_goal_sets_validation_event_type():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "verify_goal"
    function_call.args = {
        "assistant_message": "Goal looks complete",
        "goal_completed": True,
        "current_screen": "Checkout summary",
        "evidence": "All requested items are visible",
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.metadata.get("event_type") == "validation"
    assert result.metadata.get("tool_name") == "verify_goal"


def test_parse_execute_ui_validate_sets_validation_event_type():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "execute_ui"
    function_call.args = {
        "assistant_message": "Validated the confirmation banner is visible",
        "action": {
            "action_type": "validate",
            "rationale": "Explicit user-requested validation",
            "is_valid": True,
            "target_name": "Order confirmation banner",
        },
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.action.action_type.value == "validate"
    assert result.metadata.get("event_type") == "validation"
    assert result.metadata.get("tool_name") == "execute_ui"


def test_parse_execute_ui_overlay_sets_condition_for_conditional_export():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "execute_ui"
    function_call.args = {
        "assistant_message": "Dismissing promo overlay",
        "action": {
            "action_type": "tap",
            "rationale": "Close overlay before continuing",
            "is_valid": True,
            "target_name": "Got It! button",
            "overlay_detected": True,
        },
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.action.action_type.value == "tap"
    assert result.action.overlay_detected is True
    assert result.action.condition == "Overlay is visible"


def test_parse_multiple_primary_execute_ui_prefers_valid_action():
    parser = ToolResponseParser()

    invalid_call = MagicMock()
    invalid_call.name = "execute_ui"
    invalid_call.args = {
        "assistant_message": "Try tapping missing app icon",
        "action": {
            "action_type": "tap",
            "rationale": "Open app",
            "is_valid": False,
            "validation_reason": "Instacart app is not visible on the current screen.",
            "target_name": "Instacart icon",
        },
    }

    valid_call = MagicMock()
    valid_call.name = "execute_ui"
    valid_call.args = {
        "assistant_message": "Tap visible search bar",
        "action": {
            "action_type": "tap",
            "rationale": "Begin search flow",
            "is_valid": True,
            "target_name": "search bar",
        },
    }

    mock_response = MockResponse(
        [
            MockCandidate(
                MockContent(
                    [MockPart(function_call=invalid_call), MockPart(function_call=valid_call)]
                )
            )
        ]
    )

    result = parser.parse(mock_response)

    assert result.metadata.get("tool_name") == "execute_ui"
    assert result.action.target == "search bar"
    assert result.action.is_valid is True


def test_parse_execute_ui_hybrid_delta_fields():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "execute_ui"
    function_call.args = {
        "assistant_message": "No meaningful change after scroll",
        "action": {
            "action_type": "scroll",
            "rationale": "Try to reveal more items",
            "is_valid": True,
            "target_name": "results list",
        },
        "delta_observed": False,
        "delta_reasoning": "Top and bottom anchors unchanged after swipe",
        "delta_confidence": 0.91,
        "previous_screen_summary": "Search results with cards",
        "current_screen_summary": "Same search results cards",
        "visible_anchors": ["Milk", "Eggs", "Bread"],
        "top_anchor": "Milk",
        "bottom_anchor": "Bread",
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )
    result = parser.parse(mock_response)

    assert result.gemini_delta is not None
    assert result.gemini_delta.delta_observed is False
    assert result.gemini_delta.delta_reasoning is not None
    assert result.gemini_delta.top_anchor == "Milk"
