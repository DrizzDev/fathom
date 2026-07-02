from __future__ import annotations

from typing import Iterable, Optional
from unittest.mock import AsyncMock, Mock

from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from tests.builders.screens import ScreenFixtures


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
    def context_manager(
        cls,
        *,
        inject_user_guidance_async: bool = True,
        user_guidance: Optional[Iterable[object]] = None,
    ) -> ContextManager:
        """
        Build a typed :class:`ContextManager` stub exposing the helpers planner and HITL surfaces consume.
        """

        inject = AsyncMock() if inject_user_guidance_async else Mock()
        guidance = list(user_guidance) if user_guidance is not None else []

        stub = Mock(spec=ContextManager)

        stub.clear_user_guidance = Mock()
        stub.inject_user_guidance = inject
        stub.consume_user_guidance = Mock()
        stub.clear_verifier_feedback = Mock()
        stub.get_user_guidance = Mock(return_value=guidance)

        return stub
