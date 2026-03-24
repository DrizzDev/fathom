"""
Integration tests for the LangGraph orchestration path.

These tests verify:
1. The graph state schema is well-formed.
2. The LangChain adapter correctly shims tool calls for the parser.
3. The graph nodes produce correct state transitions.
4. The tool conversion from Gemini-native to LangChain format works.
5. The routing functions make correct decisions.
6. The LangChain adapter has full caching parity with GeminiLLMClient.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fathom.constants import ActionType
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.services.parsing import ToolResponseParser

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_screen_state() -> ScreenState:
    return ScreenState(
        activity="com.example/.MainActivity",
        timestamp=1000000,
        activity_hash="abcd1234",
        structural_hash="struct1234",
        visual_hash="visual1234567890",
    )


@pytest.fixture
def sample_capture(sample_screen_state: ScreenState) -> ScreenCapture:
    return ScreenCapture(
        width=1080,
        height=2400,
        activity="com.example/.MainActivity",
        image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        timestamp=1000000,
        state=sample_screen_state,
    )


@pytest.fixture
def sample_action() -> Action:
    return Action(
        action_type=ActionType.TAP,
        rationale="Tap on the button",
        target="Submit Button",
        natural_language_target="Submit Button",
        bounds=Bounds(x=400, y=600, width=200, height=100),
        confidence=0.95,
    )


@pytest.fixture
def sample_step(sample_action: Action) -> Step:
    return Step(
        action=sample_action,
        screen_hash="abc123",
        step_number=0,
    )


@pytest.fixture
def sample_plan(sample_step: Step) -> PlanResult:
    return PlanResult(
        step=sample_step,
        is_complete=False,
        reason="Step planned",
        metadata={"tool_name": "execute_ui", "tool_args": {}},
    )


# ── 1. Graph State Tests ───────────────────────────────────────────────


class TestFathomGraphState:
    def test_state_is_typed_dict(self) -> None:
        from fathom.graph.state import FathomGraphState

        # FathomGraphState should be usable as a dict
        state: FathomGraphState = {
            "intent": "Open settings",
            "max_steps": 10,
            "use_xml": False,
            "step_number": 0,
            "is_complete": False,
            "should_retry": False,
        }
        assert state["intent"] == "Open settings"
        assert state["is_complete"] is False

    def test_state_total_false_allows_partial(self) -> None:
        """total=False means we can create state with only some fields."""
        from fathom.graph.state import FathomGraphState

        state: FathomGraphState = {"intent": "test"}
        assert state["intent"] == "test"


# ── 2. LangChain Adapter Tests ─────────────────────────────────────────


class TestLangChainAdapterShim:
    """Test that the adapter correctly translates tool calls for ToolResponseParser."""

    def test_function_call_shim(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import _FunctionCallShim

        tc = {"name": "execute_ui", "args": {"assistant_message": "test", "action": {}}}
        shim = _FunctionCallShim(tc)
        assert shim.name == "execute_ui"
        assert shim.args["assistant_message"] == "test"

    def test_shim_response_parsed_by_tool_response_parser(self) -> None:
        """Verify that the shim response object is parseable by ToolResponseParser."""
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

        tool_calls = [
            {
                "name": "execute_ui",
                "args": {
                    "assistant_message": "Tapping the button",
                    "action": {
                        "action_type": "tap",
                        "rationale": "Tap submit",
                        "is_valid": True,
                        "target_name": "Submit",
                        "bbox": {"x": 400, "y": 600, "width": 200, "height": 100},
                        "confidence": 0.9,
                    },
                },
                "id": "call_123",
            }
        ]

        # Access the static method
        shim = LangChainLLMClient._LangChainLLMClient__build_shim_response_from_tool_calls(
            tool_calls
        )
        parser = ToolResponseParser()
        result = parser.parse(shim)

        assert isinstance(result, AnalysisResult)
        assert result.action.action_type == ActionType.TAP
        assert result.action.natural_language_target == "Submit"
        assert result.is_goal_complete is False

    def test_shim_response_verify_goal(self) -> None:
        """Verify goal tool call shim."""
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

        tool_calls = [
            {
                "name": "verify_goal",
                "args": {
                    "assistant_message": "Goal is complete",
                    "goal_completed": True,
                    "current_screen": "Settings",
                    "evidence": "Settings page is visible",
                },
                "id": "call_456",
            }
        ]

        shim = LangChainLLMClient._LangChainLLMClient__build_shim_response_from_tool_calls(
            tool_calls
        )
        parser = ToolResponseParser()
        result = parser.parse(shim)

        assert result.is_goal_complete is True
        assert result.action.action_type == ActionType.COMPLETE
        assert result.metadata.get("event_type") == "validation"

    def test_shim_text_fallback(self) -> None:
        """When no tool calls, text fallback should produce a valid result."""
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

        shim = LangChainLLMClient._LangChainLLMClient__build_shim_response("No actions possible")
        parser = ToolResponseParser()
        result = parser.parse(shim)

        assert isinstance(result, AnalysisResult)
        assert result.action.action_type == ActionType.WAIT


# ── 3. Tool Conversion Tests ───────────────────────────────────────────


class TestToolConversion:
    """Test conversion from Gemini-native tool definitions to LangChain format."""

    def test_convert_tools_produces_openai_format(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient
        from fathom.tools.definitions import ToolRegistry

        gemini_tools = ToolRegistry.get_all_definitions()

        # Access the static method
        lc_tools = LangChainLLMClient._LangChainLLMClient__convert_tools(gemini_tools)

        assert len(lc_tools) == 6
        for tool in lc_tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_converted_execute_ui_has_correct_properties(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient
        from fathom.tools.definitions import ToolRegistry

        gemini_tools = ToolRegistry.get_all_definitions()
        lc_tools = LangChainLLMClient._LangChainLLMClient__convert_tools(gemini_tools)

        execute_ui = next(t for t in lc_tools if t["function"]["name"] == "execute_ui")
        params = execute_ui["function"]["parameters"]

        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "assistant_message" in params["properties"]

    def test_type_mapping(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient
        from fathom.tools.definitions import ToolRegistry

        gemini_tools = ToolRegistry.get_all_definitions()
        lc_tools = LangChainLLMClient._LangChainLLMClient__convert_tools(gemini_tools)

        store_mem = next(t for t in lc_tools if t["function"]["name"] == "store_memory")
        params = store_mem["function"]["parameters"]

        assert params["properties"]["category"]["type"] == "string"
        assert params["properties"]["item"]["type"] == "string"
        assert params["properties"]["value"]["type"] == "string"


# ── 4. LangChain Tool Schema Tests ─────────────────────────────────────


class TestLangChainTools:
    """Test the @tool-decorated functions in fathom.graph.tools."""

    def test_all_tools_registered(self) -> None:
        from fathom.graph.tools import ALL_TOOLS

        names = {t.name for t in ALL_TOOLS}
        assert names == {
            "execute_ui",
            "complete_goal",
            "validate_state",
            "verify_goal",
            "store_memory",
            "recall_memory",
        }

    def test_get_tools_for_mode_default(self) -> None:
        from fathom.graph.tools import get_tools_for_mode

        tools = get_tools_for_mode("default")
        names = {t.name for t in tools}
        assert "execute_ui" in names
        assert "store_memory" in names
        assert "recall_memory" in names
        assert "verify_goal" in names

    def test_get_tools_for_mode_discovery(self) -> None:
        from fathom.graph.tools import get_tools_for_mode

        tools = get_tools_for_mode("discovery")
        names = {t.name for t in tools}
        assert names == {"execute_ui", "complete_goal", "store_memory"}

    def test_get_tools_for_mode_verification(self) -> None:
        from fathom.graph.tools import get_tools_for_mode

        tools = get_tools_for_mode("verification")
        names = {t.name for t in tools}
        assert "verify_goal" in names
        assert "recall_memory" in names
        assert "complete_goal" in names


# ── 5. Routing Function Tests ──────────────────────────────────────────


class TestRoutingFunctions:
    """Test the conditional edge routing logic."""

    def _make_ctx(self) -> Any:
        """Create a minimal NodeContext mock."""
        from fathom.agent.state import AgentState

        ctx = MagicMock()
        ctx.agent_state = AgentState(intent="test", max_steps=10)
        ctx.audit_service = MagicMock()
        ctx.is_cancelled = False
        return ctx

    def test_route_after_ground_capture_failed(self) -> None:
        from fathom.graph.nodes import make_route_after_ground

        ctx = self._make_ctx()
        route = make_route_after_ground(ctx)

        state = {"capture": None}
        assert route(state) == "done"

    def test_route_after_ground_capture_ok(self) -> None:
        from fathom.graph.nodes import make_route_after_ground

        ctx = self._make_ctx()
        route = make_route_after_ground(ctx)

        state = {"capture": MagicMock()}
        assert route(state) == "validate"

    def test_route_after_analyze_complete_no_step(self) -> None:
        from fathom.graph.nodes import make_route_after_analyze

        ctx = self._make_ctx()
        route = make_route_after_analyze(ctx)

        state = {"is_complete": True, "planned_step": None, "should_retry": False}
        assert route(state) == "done"

    def test_route_after_analyze_complete_with_step(self) -> None:
        from fathom.graph.nodes import make_route_after_analyze

        ctx = self._make_ctx()
        route = make_route_after_analyze(ctx)

        state = {"is_complete": True, "planned_step": MagicMock(), "should_retry": False}
        assert route(state) == "resolve"

    def test_route_after_analyze_retry(self) -> None:
        from fathom.graph.nodes import make_route_after_analyze

        ctx = self._make_ctx()
        route = make_route_after_analyze(ctx)

        state = {"is_complete": False, "planned_step": None, "should_retry": True}
        assert route(state) == "ground"

    def test_route_after_analyze_normal_step(self) -> None:
        from fathom.graph.nodes import make_route_after_analyze

        ctx = self._make_ctx()
        route = make_route_after_analyze(ctx)

        state = {"is_complete": False, "planned_step": MagicMock(), "should_retry": False}
        assert route(state) == "resolve"

    def test_route_after_record_complete(self) -> None:
        from fathom.graph.nodes import make_route_after_record

        ctx = self._make_ctx()
        ctx.agent_state.mark_complete(reason="done")
        route = make_route_after_record(ctx)

        state = {"is_complete": True}
        assert route(state) == "done"

    def test_route_after_record_continue(self) -> None:
        from fathom.graph.nodes import make_route_after_record

        ctx = self._make_ctx()
        route = make_route_after_record(ctx)

        state = {"is_complete": False}
        assert route(state) == "ground"


class TestValidationRequestGuards:
    def test_intent_requests_validation_keyword_detection(self) -> None:
        from fathom.graph.nodes import _intent_requests_validation

        assert _intent_requests_validation("validate the logged in state") is True
        assert _intent_requests_validation("verify checkout summary is visible") is True
        assert _intent_requests_validation("open cart and continue") is False

    def test_has_validation_event_detects_validate_steps(self) -> None:
        from fathom.graph.nodes import _has_validation_event
        from fathom.schemas.steps import StepResult

        action = Action(
            action_type=ActionType.VALIDATE,
            rationale="Validate banner",
            target="banner",
            natural_language_target="banner",
        )
        step = Step(
            action=action,
            screen_hash="hash1",
            step_number=1,
            event_type="validation",
        )
        result = StepResult(
            step=step,
            success=True,
            pre_hash="pre",
            post_hash="post",
            screen_changed=False,
            duration=10,
        )

        assert _has_validation_event([result]) is True

    def test_extract_validation_target_prefers_required_items(self) -> None:
        from fathom.graph.nodes import _extract_validation_target

        target = _extract_validation_target(
            "validate that lemon has a price",
            required_items=["lemon item with price", "logged-in indicator"],
        )
        assert target == "lemon item with price, logged-in indicator"

    def test_extract_validation_target_from_intent_phrase(self) -> None:
        from fathom.graph.nodes import _extract_validation_target

        target = _extract_validation_target(
            "Open app and validate that the user is logged in, then continue"
        )
        assert "user is logged in" in target

    def test_should_force_pre_action_validation_for_interactive_steps(self) -> None:
        from fathom.graph.nodes import _should_force_pre_action_validation

        action = Action(
            action_type=ActionType.TAP,
            rationale="Tap search",
            target="search bar",
            natural_language_target="search bar",
        )
        step = Step(action=action, screen_hash="h1", step_number=2)
        should_force = _should_force_pre_action_validation(
            intent="validate login then search lemon",
            already_validated=False,
            step_count=2,
            planned_step=step,
        )
        assert should_force is True

    def test_should_not_force_pre_action_validation_on_first_step_or_wait(self) -> None:
        from fathom.graph.nodes import _should_force_pre_action_validation

        wait_step = Step(
            action=Action(
                action_type=ActionType.WAIT,
                rationale="Wait for app load",
                target="loading",
                natural_language_target="loading",
            ),
            screen_hash="h2",
            step_number=1,
        )
        assert (
            _should_force_pre_action_validation(
                intent="validate login then search lemon",
                already_validated=False,
                step_count=0,
                planned_step=wait_step,
            )
            is False
        )
        assert (
            _should_force_pre_action_validation(
                intent="validate login then search lemon",
                already_validated=False,
                step_count=1,
                planned_step=wait_step,
            )
            is False
        )

    def test_is_validation_step_helper(self) -> None:
        from fathom.graph.nodes import (
            _is_validation_step,
            _last_step_was_validation,
            _trailing_validation_count,
        )
        from fathom.schemas.steps import StepResult

        validate_step = Step(
            action=Action(
                action_type=ActionType.VALIDATE,
                rationale="Validate login",
                target="profile icon",
                natural_language_target="profile icon",
            ),
            screen_hash="v1",
            step_number=1,
            event_type="validation",
        )
        tap_step = Step(
            action=Action(
                action_type=ActionType.TAP,
                rationale="Tap search",
                target="search",
                natural_language_target="search",
            ),
            screen_hash="a1",
            step_number=2,
        )

        assert _is_validation_step(validate_step) is True
        assert _is_validation_step(tap_step) is False
        validate_result = StepResult(
            step=validate_step,
            success=True,
            pre_hash="p",
            post_hash="q",
            screen_changed=False,
            duration=11,
        )
        tap_result = StepResult(
            step=tap_step,
            success=True,
            pre_hash="p",
            post_hash="q",
            screen_changed=True,
            duration=12,
        )
        assert _last_step_was_validation([validate_result]) is True
        assert _last_step_was_validation([validate_result, tap_result]) is False
        assert _trailing_validation_count([validate_result]) == 1
        assert _trailing_validation_count([validate_result, tap_result]) == 0
        assert _trailing_validation_count([tap_result, validate_result, validate_result]) == 2

    def test_overlay_condition_guard_detects_missing_condition(self) -> None:
        from fathom.graph.nodes import _is_overlay_step_missing_condition

        overlay_step = Step(
            action=Action(
                action_type=ActionType.TAP,
                rationale="Dismiss promotional overlay",
                target="Got It! button",
                natural_language_target="Got It! button",
                overlay_detected=True,
            ),
            screen_hash="ov1",
            step_number=1,
        )
        assert _is_overlay_step_missing_condition(overlay_step) is True

    def test_overlay_condition_guard_allows_guarded_overlay_step(self) -> None:
        from fathom.graph.nodes import _is_overlay_step_missing_condition

        overlay_step = Step(
            action=Action(
                action_type=ActionType.TAP,
                rationale="Dismiss promotional overlay",
                target="Got It! button",
                natural_language_target="Got It! button",
                overlay_detected=True,
                condition="Promotional overlay is visible",
            ),
            screen_hash="ov2",
            step_number=1,
        )
        assert _is_overlay_step_missing_condition(overlay_step) is False

    def test_overlay_condition_fallback_autofills_missing_condition(self) -> None:
        from fathom.graph.nodes import _apply_overlay_condition_fallback

        overlay_step = Step(
            action=Action(
                action_type=ActionType.TAP,
                rationale="Dismiss promotional overlay",
                target="Got It! button",
                natural_language_target="Got It! button",
                overlay_detected=True,
            ),
            screen_hash="ov3",
            step_number=1,
        )

        patched = _apply_overlay_condition_fallback(overlay_step)
        assert patched is not None
        assert patched.action.condition == "Promotional overlay is visible"


# ── 6. Settings Feature Flag Test ──────────────────────────────────────


class TestFeatureFlag:
    def test_use_langgraph_default_true(self) -> None:
        from fathom.settings.env import FathomSettings

        settings = FathomSettings()
        assert settings.use_langgraph is True

    def test_use_langgraph_env_override_false(self) -> None:
        from fathom.settings.env import FathomSettings

        with patch.dict("os.environ", {"USE_LANGGRAPH": "false"}):
            settings = FathomSettings()
            assert settings.use_langgraph is False


class TestValidationEventRecord:
    def test_step_result_to_record_includes_validation_event_type(self) -> None:
        action = Action(
            action_type=ActionType.WAIT,
            rationale="Validate loading transient state",
            target="State Validation",
            natural_language_target="State Validation",
        )
        step = Step(
            action=action,
            screen_hash="abc123",
            step_number=1,
            event_type="validation",
        )
        step_result = StepResult(
            step=step,
            success=True,
            pre_hash="pre",
            post_hash="post",
            screen_changed=False,
            duration=123,
        )

        record = step_result.to_record().model_dump()
        assert record["event_type"] == "validation"


class TestPlannerValidationEventPropagation:
    @pytest.mark.asyncio
    async def test_completion_path_preserves_validation_event_type(self) -> None:
        from fathom.agent.planner import StepPlanner
        from fathom.agent.reasoner import Reasoner
        from fathom.agent.state import AgentState

        capture = ScreenCapture(
            width=1080,
            height=2400,
            activity="com.example/.MainActivity",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
            timestamp=1000000,
            state=ScreenState(
                activity="com.example/.MainActivity",
                timestamp=1000000,
                activity_hash="abcd1234",
                structural_hash="struct1234",
                visual_hash="visual1234567890",
            ),
        )
        analysis = AnalysisResult(
            action=Action(
                action_type=ActionType.VALIDATE,
                rationale="Validation requested",
                target="profile icon",
                natural_language_target="profile icon",
                confidence=1.0,
            ),
            alternatives=[],
            reasoning="Validation succeeded",
            screen_description="Home screen",
            is_goal_complete=True,
            metadata={"event_type": "validation"},
        )

        vision = MagicMock()
        vision.analyze = AsyncMock(return_value=analysis)
        planner = StepPlanner(vision_tool=vision)
        state = AgentState(intent="validate login", max_steps=5)
        reasoner = Reasoner(intent="validate login")

        plan = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=capture,
        )

        assert plan.step is not None
        assert plan.step.event_type == "validation"


class TestNoXmlDeltaHelpers:
    def test_compute_no_xml_delta_marks_activity_change(self) -> None:
        from fathom.graph.nodes import _compute_no_xml_delta

        pre = ScreenState(
            activity="com.example/.A",
            timestamp=1000,
            activity_hash="aaaa1111",
            structural_hash="struct",
            visual_hash="11112222",
        )
        post = ScreenState(
            activity="com.example/.B",
            timestamp=1001,
            activity_hash="bbbb2222",
            structural_hash="struct",
            visual_hash="11112222",
        )

        delta = _compute_no_xml_delta(pre_state=pre, post_state=post)
        assert delta.changed is True
        assert delta.reason == "activity_changed"
        assert delta.delta_score == 1.0

    def test_compute_no_xml_delta_marks_visual_unchanged(self) -> None:
        from fathom.graph.nodes import _compute_no_xml_delta

        pre = ScreenState(
            activity="com.example/.A",
            timestamp=1000,
            activity_hash="aaaa1111",
            structural_hash="struct",
            visual_hash="11112222",
        )
        post = ScreenState(
            activity="com.example/.A",
            timestamp=1001,
            activity_hash="aaaa1111",
            structural_hash="struct",
            visual_hash="11112222",
        )

        delta = _compute_no_xml_delta(pre_state=pre, post_state=post)
        assert delta.changed is False
        assert delta.reason == "visual_unchanged"


# ── 7. MIME Detection Test ──────────────────────────────────────────────


class TestMimeDetection:
    def test_png(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        mime = LangChainLLMClient._LangChainLLMClient__detect_mime(data)
        assert mime == "image/png"

    def test_jpeg(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

        data = b"\xff\xd8\xff" + b"\x00" * 50
        mime = LangChainLLMClient._LangChainLLMClient__detect_mime(data)
        assert mime == "image/jpeg"

    def test_unknown_defaults_jpeg(self) -> None:
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient

        data = b"\x00\x00\x00" + b"\x00" * 50
        mime = LangChainLLMClient._LangChainLLMClient__detect_mime(data)
        assert mime == "image/jpeg"


# ── 8. LangChain Adapter Caching Parity Tests ──────────────────────────


class TestLangChainCachingParity:
    """
    Verify that the LangChain adapter uses CacheService exactly like
    GeminiLLMClient for prompt caching.
    """

    def _build_client(self) -> Any:
        """
        Build a LangChainLLMClient with everything mocked so no real API
        calls are made.
        """
        from fathom.infrastructure.llm.langchain_adapter import LangChainLLMClient
        from fathom.schemas.configuration import GeminiConfig

        config = GeminiConfig(
            api_key="test-key",
            model="gemini-2.0-flash-lite",
        )

        # mock CacheService instance
        mock_cache = MagicMock()
        mock_cache.stats = MagicMock()
        mock_cache.stats.to_dict.return_value = {
            "hits": 3,
            "misses": 1,
            "creates": 1,
            "evictions": 0,
            "hit_rate": 0.75,
        }
        mock_cache.get_cached_content = AsyncMock(return_value=None)
        mock_cache.delete_cache = AsyncMock()

        # mock LangChain model
        mock_model = MagicMock()

        with (
            patch("google.genai.Client", return_value=MagicMock()),
            patch(
                "fathom.infrastructure.llm.langchain_adapter.CacheService",
                return_value=mock_cache,
            ),
            patch(
                "langchain_google_genai.ChatGoogleGenerativeAI",
                return_value=mock_model,
            ),
        ):
            client = LangChainLLMClient(configuration=config)

        return client, mock_cache, mock_model

    def test_cache_service_initialized(self) -> None:
        """CacheService must be created during __init__."""
        client, mock_cache, _ = self._build_client()
        # Access private attribute via name mangling
        cache = client._LangChainLLMClient__cache
        assert cache is not None
        assert cache is mock_cache

    def test_cache_stats_returns_real_stats(self) -> None:
        """cache_stats property should delegate to CacheService.stats."""
        client, mock_cache, _ = self._build_client()
        stats = client.cache_stats
        assert stats["hits"] == 3
        assert stats["hit_rate"] == 0.75
        mock_cache.stats.to_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_deletes_cache(self) -> None:
        """cleanup() must call CacheService.delete_cache()."""
        client, mock_cache, _ = self._build_client()
        await client.cleanup()
        mock_cache.delete_cache.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_cache_miss_includes_system_message(self) -> None:
        """
        When CacheService returns None (miss), the system instruction
        should appear as a SystemMessage and tools should be bound.
        """
        client, mock_cache, mock_model = self._build_client()

        # Cache miss
        mock_cache.get_cached_content = AsyncMock(return_value=None)

        # Mock the model's ainvoke to return a tool call response
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "name": "execute_ui",
                "args": {
                    "assistant_message": "tap",
                    "actions": [
                        {
                            "action_type": "tap",
                            "rationale": "tap",
                            "is_valid": True,
                            "target_name": "btn",
                            "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
                            "confidence": 0.9,
                        }
                    ],
                    "goal_completed": False,
                },
                "id": "call_1",
            }
        ]
        mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 20}

        # bind_tools returns a model that has ainvoke
        bound_model = MagicMock()
        bound_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_model.bind_tools = MagicMock(return_value=bound_model)

        tools = {
            "function_declarations": [
                {
                    "name": "execute_ui",
                    "description": "Execute UI actions",
                    "parameters": {"type": "OBJECT", "properties": {}},
                }
            ]
        }

        result = await client.analyze(
            system_instruction="You are a test assistant",
            user_content=["Test prompt"],
            tools=tools,
        )

        # Verify cache was consulted
        mock_cache.get_cached_content.assert_awaited_once()

        # Verify tools were bound (no cache → bind tools)
        mock_model.bind_tools.assert_called_once()

        # Verify the messages include a SystemMessage
        call_args = bound_model.ainvoke.call_args[0][0]
        from langchain_core.messages import SystemMessage

        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "You are a test assistant"

        assert isinstance(result, AnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_cache_hit_uses_raw_genai_client(self) -> None:
        """
        When CacheService returns a cache name (hit), the raw genai.Client
        should be used directly (bypassing LangChain) to avoid cache-name
        mangling by ChatVertexAI / ChatGoogleGenerativeAI.
        """
        client, mock_cache, mock_model = self._build_client()

        # Cache hit
        cache_name = "projects/p/locations/l/cachedContents/abc123"
        mock_cache.get_cached_content = AsyncMock(return_value=cache_name)

        # Build a mock genai response that ToolResponseParser can parse
        mock_fc = MagicMock()
        mock_fc.name = "execute_ui"
        mock_fc.args = {
            "assistant_message": "tap",
            "actions": [
                {
                    "action_type": "tap",
                    "rationale": "tap",
                    "is_valid": True,
                    "target_name": "btn",
                    "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
                    "confidence": 0.9,
                }
            ],
            "goal_completed": False,
        }

        mock_part = MagicMock()
        mock_part.function_call = mock_fc
        mock_part.text = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 50
        mock_usage.candidates_token_count = 20
        mock_usage.cached_content_token_count = 40

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        # Mock the raw genai client's generate_content
        genai_client = client._LangChainLLMClient__genai_client
        genai_client.aio = MagicMock()
        genai_client.aio.models = MagicMock()
        genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        tools = {
            "function_declarations": [
                {
                    "name": "execute_ui",
                    "description": "Execute UI actions",
                    "parameters": {"type": "OBJECT", "properties": {}},
                }
            ]
        }

        result = await client.analyze(
            system_instruction="You are a test assistant",
            user_content=["Test prompt"],
            tools=tools,
        )

        # Verify cache was consulted
        mock_cache.get_cached_content.assert_awaited_once()

        # Verify LangChain model was NOT used (no bind_tools, no ainvoke)
        mock_model.bind_tools.assert_not_called()

        # Verify raw genai client WAS used
        genai_client.aio.models.generate_content.assert_awaited_once()

        # Verify cached tokens are reported
        assert result.metrics.get("cached_tokens") == 40
        assert isinstance(result, AnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_passes_function_declarations_to_cache(self) -> None:
        """
        CacheService.get_cached_content should receive the raw
        function_declarations list, matching GeminiLLMClient behaviour.
        """
        client, mock_cache, mock_model = self._build_client()
        mock_cache.get_cached_content = AsyncMock(return_value=None)

        mock_response = MagicMock()
        mock_response.tool_calls = [
            {
                "name": "verify_goal",
                "args": {
                    "assistant_message": "done",
                    "goal_completed": True,
                    "current_screen": "home",
                    "evidence": "visible",
                },
                "id": "call_2",
            }
        ]
        mock_response.usage_metadata = {}
        bound_model = MagicMock()
        bound_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_model.bind_tools = MagicMock(return_value=bound_model)

        decls = [
            {
                "name": "verify_goal",
                "description": "Verify goal completion",
                "parameters": {"type": "OBJECT", "properties": {}},
            }
        ]
        tools = {"function_declarations": decls}

        await client.analyze(
            system_instruction="sys",
            user_content=["content"],
            tools=tools,
        )

        # Verify the raw function_declarations list was passed to cache
        mock_cache.get_cached_content.assert_awaited_once_with(
            system_instruction="sys",
            tools=decls,
        )
