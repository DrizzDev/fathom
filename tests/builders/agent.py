from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Optional, cast
from unittest.mock import AsyncMock, Mock

from tests.builders.screens import ScreenFixtures

from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities


class AgentFixtures:
    """
    Factory for :class:`AgentState` and :class:`ContextManager` stubs used across
    planner, HITL, and graph-node tests.
    """

    @classmethod
    def state(
        cls,
        *,
        intent: str = "test intent",
        hitl_enabled: bool = False,
    ) -> AgentState:
        """
        Build a fresh :class:`AgentState` whose HITL capability is configurable.
        """

        return AgentState(
            intent=intent,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl_enabled)),
        )

    @classmethod
    def stuck_state(
        cls,
        *,
        action_type: str = "tap",
        intent: str = "test intent",
        hitl_enabled: bool = False,
        action_description: str = "Tap Continue",
    ) -> AgentState:
        """
        Build an :class:`AgentState` whose loop detector has been driven to stuck.
        """

        state = cls.state(intent=intent, hitl_enabled=hitl_enabled)

        screen = ScreenFixtures.state()
        detector = state.runtime.screen.detector

        for _ in range(detector.threshold):
            detector.record(
                screen=screen,
                action_type=action_type,
                action_description=action_description,
            )
        return state

    @classmethod
    def context_manager_stub(
        cls,
        *,
        inject_user_guidance_async: bool = True,
        user_guidance: Optional[Iterable[object]] = None,
    ) -> ContextManager:
        """
        Build a duck-typed :class:`ContextManager` stub exposing the helpers
        planner and HITL surfaces consume; safe to cast for typed call sites.
        """

        inject = AsyncMock() if inject_user_guidance_async else Mock()
        guidance = list(user_guidance) if user_guidance is not None else []

        stub = SimpleNamespace(
            clear_user_guidance=Mock(),
            inject_user_guidance=inject,
            consume_user_guidance=Mock(),
            clear_verifier_feedback=Mock(),
            get_user_guidance=Mock(return_value=guidance),
        )
        return cast("ContextManager", stub)
