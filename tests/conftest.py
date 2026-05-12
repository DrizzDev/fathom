from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


from fathom.interfaces.llm import LLMPort, PromptPart  # noqa: E402
from fathom.interfaces.memory import MemoryPort  # noqa: E402
from fathom.schemas.actions import Action  # noqa: E402
from fathom.schemas.conversation import ConversationTurn  # noqa: E402
from fathom.schemas.results import GenerateResult  # noqa: E402
from fathom.schemas.screens import ScreenState  # noqa: E402


class __LLMStub(LLMPort):
    """
    Deterministic :class:`LLMPort` double. ``generate`` returns an
    empty JSON object so the decomposer's parse path runs but yields no sub-goals; ``cleanup`` is a no-op.
    """

    @property
    def model_name(self) -> str:
        """
        Stable model name used by ``PromptFactory.get_decomposition_builder`` to resolve the prompt-builder variant.
        """

        return "stub-model"

    async def generate(
        self,
        *,
        use_cache: bool,
        prompt: Sequence[PromptPart],
        tools: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
    ) -> GenerateResult:
        """
        Return an empty JSON payload regardless of input so callers can
        exercise the surrounding code paths without a live model.
        """

        _ = use_cache, prompt, tools, system_instruction, conversation_history
        return GenerateResult(content="{}")

    async def cleanup(self) -> None:
        """
        No resources to release; declared to satisfy the LLMPort contract.
        """

        return None


class __MemoryStub(MemoryPort):
    """
    No-op :class:`MemoryPort` double for tests that do not exercise memory persistence.
    Every getter returns an empty value and every setter is silently dropped.
    """

    async def set(self, *, key: str, value: str) -> None:
        """
        Drop the key/value silently.
        """

        _ = key, value
        return None

    async def get(self, *, key: str) -> Optional[str]:
        """
        Always miss to simulate an empty memory store.
        """

        _ = key
        return None

    async def get_all(self) -> dict:
        """
        Return an empty key/value map.
        """

        return {}

    async def store_observation(self, *, screen: ScreenState, description: Optional[str]) -> None:
        """
        Drop the observation silently.
        """

        _ = screen, description
        return None

    async def store_experience(self, *, visual_hash: str, action: Action, success: bool) -> None:
        """
        Drop the experience silently.
        """

        _ = visual_hash, action, success
        return None

    async def retrieve_knowledge(self, *, visual_hash: str) -> dict:
        """
        Always return an empty knowledge dict to simulate a cold cache.
        """

        _ = visual_hash
        return {}

    async def get_all_knowledge(self) -> dict:
        """
        Return an empty aggregate knowledge map.
        """

        return {}


@pytest.fixture
def llm_port_stub() -> LLMPort:
    """
    Provide a deterministic :class:`LLMPort` stub for tests that need to inject one without standing up a real provider.
    """

    return __LLMStub()


@pytest.fixture
def memory_port_stub() -> MemoryPort:
    """
    Provide a no-op :class:`MemoryPort` stub for tests that do not exercise memory persistence.
    """

    return __MemoryStub()
