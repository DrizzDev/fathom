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

    function_call = MagicMock()
    function_call.name = "explore_ui"
    function_call.args = {
        "action": {
            "action_type": "swipe_up",
            "rationale": "scroll for more",
            "target_name": "content area",
        },
        "assistant_message": "Swiping up to check for more elements",
        "screen_description": "Home feed with posts",
        "content_exhausted": True,
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.content_exhausted is True


def test_parse_content_exhausted_default_false():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "explore_ui"
    function_call.args = {
        "action": {
            "action_type": "tap",
            "rationale": "P1 navigation tab",
            "target_name": "Home tab",
        },
        "assistant_message": "Tapping Home tab — untried P1 navigation",
        "screen_description": "App home screen",
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.content_exhausted is False


def test_parse_explore_ui_overlay_detected():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "explore_ui"
    function_call.args = {
        "assistant_message": "Dismissing promo overlay",
        "screen_description": "Home screen with promo popup",
        "action": {
            "action_type": "tap",
            "rationale": "Close overlay before exploring",
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
    assert result.metadata.get("tool_name") == "explore_ui"


def test_parse_explore_ui_element_category_and_expected_outcome():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "explore_ui"
    function_call.args = {
        "assistant_message": "Tapping Settings icon — P4 secondary action",
        "screen_description": "Home feed with bottom navigation",
        "action": {
            "action_type": "tap",
            "rationale": "P4 secondary action, not yet tried",
            "target_name": "Settings icon",
            "element_category": "secondary_action",
            "expected_outcome": "new_screen",
            "confidence": 0.95,
        },
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.metadata.get("element_category") == "secondary_action"
    assert result.metadata.get("expected_outcome") == "new_screen"
    assert result.action.target == "Settings icon"
    assert result.action.confidence == 0.95


def test_parse_unknown_tool_call_returns_fallback():
    parser = ToolResponseParser()

    function_call = MagicMock()
    function_call.name = "unknown_tool"
    function_call.args = {"assistant_message": "something"}

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.action.action_type.value == "wait"
    assert result.action.confidence == 0.0


def test_parse_no_function_call_returns_fallback():
    parser = ToolResponseParser()

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(text="I cannot see any elements")]))]
    )

    result = parser.parse(mock_response)

    assert result.action.action_type.value == "wait"


def test_parse_blocked_response_returns_fallback():
    parser = ToolResponseParser()

    mock_response = MockResponse([MockCandidate(MockContent([MockPart(text="blocked")]))])
    mock_response.candidates[0].finish_reason = "SAFETY"

    result = parser.parse(mock_response)

    assert result.action.action_type.value == "wait"
    assert "Content filtered" in result.reasoning
