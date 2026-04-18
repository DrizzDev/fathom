"""Tests for IntentClassifier tool-call behavior."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import pytest

from fathom.core.prompts.classification import CLASSIFICATION_TOOL_NAME
from fathom.core.services.intent_classifier import IntentClassifier
from fathom.interfaces.llm import LLMPort
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import GenerateResult


class _FakeToolCall:
    """Object-shaped tool call (mirrors the Gemini adapter response)."""

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        self.name = name
        self.args = args


class ClassifierMockLLM(LLMPort):
    """Mock LLM that returns a pre-canned GenerateResult."""

    def __init__(self, result: GenerateResult, *, raise_exc: Optional[Exception] = None) -> None:
        self._result = result
        self._raise_exc = raise_exc
        self.last_tools: Optional[Dict[str, Any]] = None
        self.last_system_instruction: Optional[str] = None
        self.last_prompt: List[Any] = []
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-classifier"

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
        self.call_count += 1
        self.last_tools = tools
        self.last_system_instruction = system_instruction
        self.last_prompt = list(prompt)
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result

    async def cleanup(self) -> None:
        pass


@pytest.mark.asyncio
async def test_classifier_returns_false_for_simple_intent() -> None:
    """Tool call with should_decompose=False must make the classifier return False."""

    tool_call = _FakeToolCall(
        name=CLASSIFICATION_TOOL_NAME,
        args={"should_decompose": False, "reason": "single login workflow"},
    )
    llm = ClassifierMockLLM(GenerateResult(content="", tool_calls=[tool_call]))
    classifier = IntentClassifier(llm=llm)

    result = await classifier.should_decompose("Log in with foo@bar.com and pass 'test'")

    assert result is False
    assert llm.call_count == 1
    assert llm.last_tools is not None
    assert "function_declarations" in llm.last_tools


@pytest.mark.asyncio
async def test_classifier_returns_true_for_complex_intent() -> None:
    """Tool call with should_decompose=True must make the classifier return True."""

    tool_call = _FakeToolCall(
        name=CLASSIFICATION_TOOL_NAME,
        args={"should_decompose": True, "reason": "requires comparing options"},
    )
    llm = ClassifierMockLLM(GenerateResult(content="", tool_calls=[tool_call]))
    classifier = IntentClassifier(llm=llm)

    result = await classifier.should_decompose(
        "Search iphone, pick the cheapest under $800, add to cart"
    )

    assert result is True


@pytest.mark.asyncio
async def test_classifier_accepts_dict_shaped_tool_call() -> None:
    """Some adapters return the tool call as a plain dict; the extractor
    must handle both shapes (mirrors LLMSummarizer)."""

    dict_tool_call: Dict[str, Any] = {
        "name": CLASSIFICATION_TOOL_NAME,
        "args": {"should_decompose": False, "reason": "single search workflow"},
    }
    llm = ClassifierMockLLM(GenerateResult(content="", tool_calls=[dict_tool_call]))
    classifier = IntentClassifier(llm=llm)

    result = await classifier.should_decompose("Search for 'iphone' on Amazon")

    assert result is False


@pytest.mark.asyncio
async def test_classifier_fails_safe_on_missing_tool_call() -> None:
    """Empty tool_calls list must default to True (decompose)."""

    llm = ClassifierMockLLM(GenerateResult(content="", tool_calls=[]))
    classifier = IntentClassifier(llm=llm)

    result = await classifier.should_decompose("any intent")

    assert result is True


@pytest.mark.asyncio
async def test_classifier_fails_safe_on_missing_should_decompose_key() -> None:
    """Tool call that omits the should_decompose arg must default to True."""

    tool_call = _FakeToolCall(
        name=CLASSIFICATION_TOOL_NAME,
        args={"reason": "forgot the bool"},
    )
    llm = ClassifierMockLLM(GenerateResult(content="", tool_calls=[tool_call]))
    classifier = IntentClassifier(llm=llm)

    result = await classifier.should_decompose("any intent")

    assert result is True


@pytest.mark.asyncio
async def test_classifier_fails_safe_on_exception() -> None:
    """Any LLM exception must default to True (decompose)."""

    llm = ClassifierMockLLM(
        GenerateResult(content="", tool_calls=[]),
        raise_exc=RuntimeError("LLM is down"),
    )
    classifier = IntentClassifier(llm=llm)

    result = await classifier.should_decompose("any intent")

    assert result is True
