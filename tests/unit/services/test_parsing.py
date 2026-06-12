"""
Unit tests for parsing Gemini tool-call responses.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock

from fathom.services.parsing import ToolResponseParser


class MockPart:
    """A single response part holding a function call or text."""

    def __init__(self, function_call: Any = None, text: Optional[str] = None) -> None:
        self.function_call = function_call
        self.text = text


class MockContent:
    """A candidate's content with ordered parts."""

    def __init__(self, parts: List[MockPart]) -> None:
        self.parts = parts


class MockCandidate:
    """A response candidate with a finish reason."""

    def __init__(self, content: MockContent) -> None:
        self.content = content
        self.finish_reason = "STOP"


class MockResponse:
    """A Gemini-style response with candidates."""

    def __init__(self, candidates: List[MockCandidate]) -> None:
        self.candidates = candidates


class TestToolResponseParser:
    """
    explore_ui tool calls are parsed into actions with threaded metadata, and
    malformed or blocked responses fall back to a safe WAIT action.
    """

    @staticmethod
    def __call(name: str, args: dict) -> MagicMock:
        function_call = MagicMock()
        function_call.name = name
        function_call.args = args
        return function_call

    @classmethod
    def __response_for(cls, function_call: MagicMock) -> MockResponse:
        return MockResponse([MockCandidate(MockContent([MockPart(function_call=function_call)]))])

    @classmethod
    def __explore(cls, action: dict, **top: Any) -> MockResponse:
        return cls.__response_for(
            cls.__call("explore_ui", {"screen_description": "screen", **top, "action": action})
        )

    def test_content_exhausted_flag(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "swipe_up",
                "rationale": "scroll for more",
                "target_name": "content area",
            },
            assistant_message="Swiping up to check for more elements",
            content_exhausted=True,
        )
        assert parser.parse(response).content_exhausted is True

    def test_content_exhausted_defaults_false(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {"action_type": "tap", "rationale": "P1 navigation tab", "target_name": "Home tab"},
            assistant_message="Tapping Home tab",
        )
        assert parser.parse(response).content_exhausted is False

    def test_overlay_detected(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "tap",
                "rationale": "Close overlay before exploring",
                "target_name": "Got It! button",
                "overlay_detected": True,
            },
            assistant_message="Dismissing promo overlay",
        )
        result = parser.parse(response)
        assert result.action.action_type.value == "tap"
        assert result.action.overlay_detected is True
        assert result.metadata.get("tool_name") == "explore_ui"

    def test_element_category_and_expected_outcome(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "tap",
                "rationale": "P4 secondary action, not yet tried",
                "target_name": "Settings icon",
                "element_category": "secondary_action",
                "expected_outcome": "new_screen",
                "confidence": 0.95,
            },
            assistant_message="Tapping Settings icon",
        )
        result = parser.parse(response)
        assert result.metadata.get("element_category") == "secondary_action"
        assert result.metadata.get("expected_outcome") == "new_screen"
        assert result.action.natural_language_target == "Settings icon"
        assert result.action.confidence == 0.95

    def test_region_enum_threaded_to_action_and_metadata(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "tap",
                "rationale": "P1 bottom-nav tab, not yet tried",
                "target_name": "Home tab",
                "element_category": "global_navigation",
                "region": "bottom_nav",
                "tap_target": {"x": 100, "y": 950},
            },
            assistant_message="Tapping Home tab in bottom nav",
        )
        result = parser.parse(response)
        assert result.metadata.get("region") == "bottom_nav"
        assert result.action.region == "bottom_nav"

    def test_type_action_threads_text(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "type",
                "rationale": "Search bar P2, untried",
                "target_name": "Search bar",
                "region": "top_bar",
                "text": "pizza",
                "tap_target": {"x": 500, "y": 100},
            },
            assistant_message="Typing 'pizza' in search",
        )
        result = parser.parse(response)
        assert result.action.action_type.value == "type"
        assert result.action.text == "pizza"

    def test_swipe_left_action_type(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "swipe_left",
                "rationale": "Horizontal carousel P4, untried",
                "target_name": "Featured restaurants carousel",
                "region": "content",
                "tap_target": {"x": 500, "y": 500},
            },
            assistant_message="Swiping carousel",
        )
        assert parser.parse(response).action.action_type.value == "swipe_left"

    def test_unknown_region_is_dropped(self) -> None:
        parser = ToolResponseParser()
        response = self.__explore(
            {
                "action_type": "tap",
                "rationale": "r",
                "target_name": "X",
                "region": "middle_left",  # not in enum
                "tap_target": {"x": 500, "y": 500},
            },
            assistant_message="msg",
        )
        assert parser.parse(response).action.region is None

    def test_unknown_tool_call_returns_fallback(self) -> None:
        parser = ToolResponseParser()
        response = self.__response_for(
            self.__call("unknown_tool", {"assistant_message": "something"})
        )
        result = parser.parse(response)
        assert result.action.action_type.value == "wait"
        assert result.action.confidence == 0.0

    def test_no_function_call_returns_fallback(self) -> None:
        parser = ToolResponseParser()
        response = MockResponse(
            [MockCandidate(MockContent([MockPart(text="I cannot see any elements")]))]
        )
        assert parser.parse(response).action.action_type.value == "wait"

    def test_blocked_response_returns_fallback(self) -> None:
        parser = ToolResponseParser()
        response = MockResponse([MockCandidate(MockContent([MockPart(text="blocked")]))])
        response.candidates[0].finish_reason = "SAFETY"
        result = parser.parse(response)
        assert result.action.action_type.value == "wait"
        assert "Content filtered" in result.reasoning
