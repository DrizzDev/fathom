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
        "actions": [{"action_type": "swipe_left", "rationale": "next item"}],
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
        "actions": [{"action_type": "swipe_left", "rationale": "next item"}],
        "assistant_message": "Swiping left",
    }

    mock_response = MockResponse(
        [MockCandidate(MockContent([MockPart(function_call=function_call)]))]
    )

    result = parser.parse(mock_response)

    assert result.content_exhausted is False
