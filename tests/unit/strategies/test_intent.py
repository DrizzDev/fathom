from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union, cast

import pytest

from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import GenerateResult
from fathom.schemas.subgoal import SubGoal
from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    IntentStrategy,
)


class IntentStrategyTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover IntentStrategy persistence setup.
    """

    async def test_build_checkpointer_context_configures_allowed_modules(self) -> None:
        """
        Build the SQLite checkpointer with the Fathom serde allowlist.
        """

        strategy = object.__new__(IntentStrategy)
        context_builder = cast(
            "Callable[[Path], Any]",
            strategy.__getattribute__("_IntentStrategy__build_checkpointer_context"),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoints.db"

            async with context_builder(checkpoint_path) as checkpointer:
                self.assertEqual(type(checkpointer).__name__, "AsyncSqliteSaver")
                self.assertEqual(type(checkpointer.serde).__name__, "JsonPlusSerializer")
                allowed_modules = getattr(
                    checkpointer.serde,
                    "_allowed_json_modules",
                    getattr(checkpointer.serde, "_allowed_modules", None),
                )
                self.assertEqual(allowed_modules, set(CHECKPOINT_ALLOWED_JSON_MODULES))
                if hasattr(checkpointer.serde, "_allowed_msgpack_modules"):
                    self.assertEqual(
                        checkpointer.serde._allowed_msgpack_modules,
                        set(CHECKPOINT_ALLOWED_MSGPACK_MODULES),
                    )


class _GateLLM(LLMPort):
    """LLM mock that returns a canned GenerateResult for every call.

    The same instance is used for both the classifier call (which
    expects a tool_calls payload) and the decomposer call (which expects
    a JSON ``content`` payload). Tests pick which one triggers by
    controlling the classifier decision.
    """

    def __init__(
        self,
        *,
        tool_call_args: Optional[Dict[str, Any]] = None,
        decomposer_json: str = '{"sub_goals": ["step 1", "step 2"], "confidence": 0.9}',
    ) -> None:
        self._tool_call_args = tool_call_args
        self._decomposer_json = decomposer_json
        self.generate_calls: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "mock-gate"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[Union[str, bytes, Dict[str, str]]],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        thinking_level: Optional[str] = None,
    ) -> GenerateResult:
        self.generate_calls.append(
            {
                "tools": tools,
                "system_instruction": system_instruction,
                "prompt": list(prompt),
            }
        )
        if tools is not None:
            # Classifier path: return a structured tool call.
            tool_call = SimpleNamespace(
                name="classify_intent",
                args=self._tool_call_args or {"should_decompose": True, "reason": "default"},
            )
            return GenerateResult(content="", tool_calls=[tool_call])
        # Decomposer path: return a JSON payload.
        return GenerateResult(content=self._decomposer_json, tool_calls=[])

    async def cleanup(self) -> None:
        pass


class _NullTelemetry:
    """Silent TelemetryPort stub that swallows every call."""

    async def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _make_strategy_stub(intent: str, llm: LLMPort) -> IntentStrategy:
    """Build a bare IntentStrategy with only the attributes
    ``__resolve_initial_sub_goals`` needs. Avoids the real constructor,
    which requires the full graph-context dependency tree."""

    strategy = object.__new__(IntentStrategy)
    # Name-mangled attribute assignments (double-underscore → _IntentStrategy__...).
    strategy._IntentStrategy__intent = intent  # type: ignore[attr-defined]
    strategy._IntentStrategy__llm = llm  # type: ignore[attr-defined]
    strategy._IntentStrategy__graph_context = SimpleNamespace(  # type: ignore[attr-defined]
        configuration=SimpleNamespace(llm=LLMConfiguration()),
        telemetry=_NullTelemetry(),
    )
    return strategy


async def _call_resolve(strategy: IntentStrategy) -> List[SubGoal]:
    resolver = cast(
        "Callable[[], Awaitable[List[SubGoal]]]",
        strategy.__getattribute__("_IntentStrategy__resolve_initial_sub_goals"),
    )
    return await resolver()


@pytest.fixture(autouse=True)
def _register_prompt_builders_for_gate_tests() -> None:
    """Decomposer path needs the PromptFactory builders registered."""

    from fathom.runtime.bootstrap import register_default_prompt_builders

    register_default_prompt_builders()


@pytest.mark.asyncio
async def test_resolve_sub_goals_simple_skips_decomposer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classifier says should_decompose=False → strategy produces a
    single SubGoal whose description is the verbatim intent, and the
    decomposer JSON branch is never hit (only one LLM call total)."""

    monkeypatch.setenv("FATHOM_ALLOW_ATOMIC_SINGLE_SUBGOAL", "true")

    llm = _GateLLM(tool_call_args={"should_decompose": False, "reason": "simple"})
    intent = "Open Instacart, go to the Aldi store page, and add banana to cart"
    strategy = _make_strategy_stub(intent=intent, llm=llm)

    sub_goals = await _call_resolve(strategy)

    assert len(sub_goals) == 1
    assert sub_goals[0].index == 0
    assert sub_goals[0].description == intent
    # Only the classifier call (with tools). No decomposer call.
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["tools"] is not None


@pytest.mark.asyncio
async def test_resolve_sub_goals_complex_runs_decomposer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classifier says should_decompose=True → strategy calls the
    decomposer (second LLM call, no tools) and returns its sub_goals."""

    monkeypatch.setenv("FATHOM_ALLOW_ATOMIC_SINGLE_SUBGOAL", "true")

    llm = _GateLLM(
        tool_call_args={"should_decompose": True, "reason": "multi-step"},
        decomposer_json='{"sub_goals": ["find iphone", "add to cart"], "confidence": 0.9}',
    )
    strategy = _make_strategy_stub(
        intent="Find the cheapest iphone on Amazon and add it to cart",
        llm=llm,
    )

    sub_goals = await _call_resolve(strategy)

    assert [sg.description for sg in sub_goals] == ["find iphone", "add to cart"]
    # Classifier (with tools) + decomposer (no tools).
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[0]["tools"] is not None
    assert llm.generate_calls[1]["tools"] is None


@pytest.mark.asyncio
async def test_resolve_sub_goals_disabled_skips_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When FATHOM_ALLOW_ATOMIC_SINGLE_SUBGOAL=false the classifier is
    never constructed and the decomposer runs on the first call
    (identifiable by tools=None on the first generate() call)."""

    monkeypatch.setenv("FATHOM_ALLOW_ATOMIC_SINGLE_SUBGOAL", "false")

    llm = _GateLLM(
        decomposer_json='{"sub_goals": ["step A", "step B"], "confidence": 0.9}',
    )
    strategy = _make_strategy_stub(intent="any intent", llm=llm)

    sub_goals = await _call_resolve(strategy)

    assert [sg.description for sg in sub_goals] == ["step A", "step B"]
    assert len(llm.generate_calls) == 1
    # The single call is the decomposer, not the classifier.
    assert llm.generate_calls[0]["tools"] is None
