"""Tests for IntentDecomposer screenshot parameter."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Union

import pytest

from fathom.core.services.decomposer import IntentDecomposer
from fathom.interfaces.llm import LLMPort
from fathom.runtime.bootstrap import register_default_prompt_builders
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import GenerateResult


@pytest.fixture(autouse=True)
def _register_prompt_builders() -> None:
    """Ensure the PromptFactory has its default builders registered.

    The production composition roots (CLI, Temporal worker) call
    ``register_default_prompt_builders`` during startup. Tests that
    instantiate services directly must do the same.
    """

    register_default_prompt_builders()


class MockLLM(LLMPort):
    """Minimal LLM mock that captures generate() arguments."""

    def __init__(self, response_json: Dict[str, Any]) -> None:
        self._response = json.dumps(response_json)
        self.last_prompt: list[Any] = []
        self.last_system_instruction: Optional[str] = None

    @property
    def model_name(self) -> str:
        return "mock-model"

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
        self.last_prompt = list(prompt)
        self.last_system_instruction = system_instruction
        return GenerateResult(content=self._response)

    async def cleanup(self) -> None:
        pass


@pytest.mark.asyncio
async def test_decompose_without_screenshot() -> None:
    """Initial decomposition should not include screenshot context."""

    llm = MockLLM({"sub_goals": ["step 1", "step 2"], "confidence": 0.9})
    decomposer = IntentDecomposer(llm=llm)

    result = await decomposer.decompose("open app and do stuff")

    assert len(result) == 2
    assert "screenshot" not in (llm.last_system_instruction or "").lower()
    assert all(isinstance(p, str) for p in llm.last_prompt)


@pytest.mark.asyncio
async def test_decompose_with_screenshot() -> None:
    """Replanning decomposition should include screenshot in prompt."""

    llm = MockLLM({"sub_goals": ["tap button", "verify"], "confidence": 0.85})
    decomposer = IntentDecomposer(llm=llm)
    fake_image = b"\x89PNG\r\n\x1a\nfake_image_data"

    result = await decomposer.decompose("tap button and verify", screenshot=fake_image)

    assert len(result) == 2
    assert "screenshot" in (llm.last_system_instruction or "").lower()
    assert "already here" in (llm.last_system_instruction or "").lower()
    assert any(isinstance(p, bytes) for p in llm.last_prompt)
