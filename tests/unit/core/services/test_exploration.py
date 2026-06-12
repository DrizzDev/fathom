from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

from fathom.core.services.exploration import ExplorationResponseParser, ExplorationVisionService
from fathom.schemas.results import GenerateResult


class _Call:
    """Minimal tool-call stand-in exposing name and args."""

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        self.name = name
        self.args = args


class TestExplorationResponseParser(unittest.TestCase):
    """explore_ui parses into an action; describe_screen attaches a rich description."""

    def setUp(self) -> None:
        self.__parser = ExplorationResponseParser()

    @staticmethod
    def __explore(*, action: Dict[str, Any], **top: Any) -> GenerateResult:
        return GenerateResult(tool_calls=[_Call("explore_ui", {"action": action, **top})])

    def test_parses_explore_ui_action(self) -> None:
        response = self.__explore(
            action={
                "action_type": "tap",
                "rationale": "P1 navigation",
                "target_name": "Home tab",
                "tap_target": {"x": 100, "y": 950},
                "element_category": "global_navigation",
                "region": "bottom_nav",
            },
            assistant_message="Tapping Home tab",
            screen_description="Home feed",
            content_exhausted=False,
        )

        result = self.__parser.parse(response)

        self.assertEqual(result.action.action_type.value, "tap")
        self.assertEqual(result.action.natural_language_target, "Home tab")
        self.assertEqual(result.action.region, "bottom_nav")
        self.assertEqual(result.action.element_category, "global_navigation")
        self.assertEqual(result.action.bounds.x, 100)  # type: ignore[union-attr]
        self.assertFalse(result.content_exhausted)
        self.assertEqual(result.metadata["region"], "bottom_nav")

    def test_content_exhausted_flag(self) -> None:
        response = self.__explore(
            action={
                "action_type": "swipe_up",
                "rationale": "scroll for more",
                "target_name": "feed",
                "tap_target": {"x": 500, "y": 500},
            },
            assistant_message="scrolling",
            content_exhausted=True,
        )

        self.assertTrue(self.__parser.parse(response).content_exhausted)

    def test_unknown_action_type_falls_back_to_tap(self) -> None:
        response = self.__explore(
            action={
                "action_type": "frobnicate",
                "rationale": "r",
                "target_name": "x",
                "tap_target": {"x": 1, "y": 1},
            },
            assistant_message="m",
        )

        self.assertEqual(self.__parser.parse(response).action.action_type.value, "tap")

    def test_unknown_region_is_dropped(self) -> None:
        response = self.__explore(
            action={
                "action_type": "tap",
                "rationale": "r",
                "target_name": "x",
                "region": "middle_left",
                "tap_target": {"x": 1, "y": 1},
            },
            assistant_message="m",
        )

        self.assertIsNone(self.__parser.parse(response).action.region)

    def test_attaches_rich_description_from_describe_screen(self) -> None:
        response = GenerateResult(
            tool_calls=[
                _Call(
                    "explore_ui",
                    {
                        "action": {
                            "action_type": "tap",
                            "rationale": "r",
                            "target_name": "Home",
                        },
                        "assistant_message": "m",
                        "screen_description": "s",
                    },
                ),
                _Call(
                    "describe_screen",
                    {
                        "activity_name": "com.x/.Home",
                        "screen_purpose": "Home",
                        "elements": "Top bar: Cart - opens cart",
                        "achievable_actions": "Search",
                    },
                ),
            ]
        )

        rich = self.__parser.parse(response).metadata["rich_description"]

        self.assertIn("## Elements", rich)
        self.assertIn("## What You Can Do", rich)

    def test_no_tool_calls_returns_wait(self) -> None:
        result = self.__parser.parse(GenerateResult(content="I cannot see elements", tool_calls=[]))

        self.assertEqual(result.action.action_type.value, "wait")

    def test_missing_action_returns_wait(self) -> None:
        response = GenerateResult(tool_calls=[_Call("explore_ui", {"assistant_message": "none"})])

        self.assertEqual(self.__parser.parse(response).action.action_type.value, "wait")


class TestExplorationVisionServiceScan(unittest.IsolatedAsyncioTestCase):
    """The scan call assembles the prompt and threads dedup feedback through."""

    @staticmethod
    def __service() -> tuple[ExplorationVisionService, AsyncMock]:
        response = GenerateResult(
            tool_calls=[
                _Call(
                    "explore_ui",
                    {
                        "action": {"action_type": "tap", "rationale": "r", "target_name": "Home"},
                        "assistant_message": "m",
                        "screen_description": "s",
                    },
                )
            ]
        )
        generate = AsyncMock(return_value=response)
        llm = Mock(generate=generate)
        return ExplorationVisionService(llm=llm), generate

    async def test_scan_prompt_carries_context_and_image(self) -> None:
        service, generate = self.__service()
        capture = Mock(image=b"img")

        result = await service.scan(capture=capture, knowledge_context="CONTEXT")

        self.assertEqual(result.action.natural_language_target, "Home")
        prompt = generate.await_args.kwargs["prompt"]
        self.assertEqual(prompt, ["CONTEXT", b"img"])

    async def test_failures_are_prepended_as_a_corrective_directive(self) -> None:
        service, generate = self.__service()
        capture = Mock(image=b"img")

        await service.scan(
            capture=capture,
            knowledge_context="CONTEXT",
            failures=['You already tried tap "Home"', "   "],
        )

        prompt = generate.await_args.kwargs["prompt"]
        self.assertEqual(len(prompt), 3)
        self.assertIn("REJECTED", prompt[0])
        self.assertIn('You already tried tap "Home"', prompt[0])
        self.assertEqual(prompt[1:], ["CONTEXT", b"img"])

    async def test_blank_failures_do_not_add_a_prompt_part(self) -> None:
        service, generate = self.__service()
        capture = Mock(image=b"img")

        await service.scan(capture=capture, knowledge_context="CONTEXT", failures=["", "  "])

        self.assertEqual(generate.await_args.kwargs["prompt"], ["CONTEXT", b"img"])
