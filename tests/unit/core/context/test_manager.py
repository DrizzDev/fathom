"""
Unit tests for :class:`ContextManager` typed feedback channels. Pins
that user guidance and verifier feedback do not pollute each other and
that the planner-prompt payload exposes both independently.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from fathom.core.context.manager import ContextManager
from fathom.schemas.feedback import UserGuidance, VerifierFeedback


class TestContextManagerChannels:
    """
    Behavioral pins for the two-channel ContextManager API.
    """

    @pytest.fixture
    async def manager(self, memory_port_stub) -> AsyncGenerator[ContextManager, None]:
        """
        Build a :class:`ContextManager` against the no-op memory stub.
        Requires a running event loop because the manager spawns a
        background persistence worker on construction.
        """

        instance = ContextManager(memory=memory_port_stub, workflow_id="test")

        try:
            yield instance
        finally:
            await instance.shutdown()

    @pytest.mark.asyncio
    async def test_user_guidance_does_not_appear_in_verifier_channel(
        self, manager: ContextManager
    ) -> None:
        """
        Injecting user guidance must not leak into the verifier-feedback
        channel.
        """

        await manager.inject_user_guidance(guidance="please dismiss the banner")

        assert manager.get_verifier_feedback() == []
        assert len(manager.get_user_guidance()) == 1

    @pytest.mark.asyncio
    async def test_verifier_feedback_does_not_appear_in_user_channel(
        self, manager: ContextManager
    ) -> None:
        """
        Injecting verifier feedback must not leak into the user-guidance
        channel.
        """

        await manager.inject_verifier_feedback(feedback="completion claim rejected")

        assert manager.get_user_guidance() == []
        assert len(manager.get_verifier_feedback()) == 1

    @pytest.mark.asyncio
    async def test_get_full_context_exposes_both_channels(self, manager: ContextManager) -> None:
        """
        ``get_full_context`` must surface both channels under their own
        keys for the prompt builder.
        """

        await manager.inject_user_guidance(guidance="use the search bar")
        await manager.inject_verifier_feedback(feedback="not on the SRP yet")

        context = manager.get_full_context()
        assert context["guidance"] == ["use the search bar"]
        assert context["verifier_feedback"] == ["not on the SRP yet"]

    @pytest.mark.asyncio
    async def test_clear_user_guidance_leaves_verifier_intact(
        self, manager: ContextManager
    ) -> None:
        """
        Clearing user guidance must not drop verifier feedback.
        """

        await manager.inject_user_guidance(guidance="A")
        await manager.inject_verifier_feedback(feedback="B")

        manager.clear_user_guidance()
        assert manager.get_user_guidance() == []
        assert len(manager.get_verifier_feedback()) == 1

    @pytest.mark.asyncio
    async def test_clear_verifier_feedback_leaves_user_intact(
        self, manager: ContextManager
    ) -> None:
        """
        Clearing verifier feedback must not drop user guidance.
        """

        await manager.inject_user_guidance(guidance="A")
        await manager.inject_verifier_feedback(feedback="B")

        manager.clear_verifier_feedback()
        assert len(manager.get_user_guidance()) == 1
        assert manager.get_verifier_feedback() == []

    @pytest.mark.asyncio
    async def test_entries_carry_correct_subtype(self, manager: ContextManager) -> None:
        """
        Each channel's entries must be of its declared schema type.
        """

        await manager.inject_user_guidance(guidance="A", step=3)
        await manager.inject_verifier_feedback(feedback="B", step=7)

        user_entries = manager.get_user_guidance()
        verifier_entries = manager.get_verifier_feedback()

        assert all(isinstance(entry, UserGuidance) for entry in user_entries)
        assert all(isinstance(entry, VerifierFeedback) for entry in verifier_entries)

        assert user_entries[0].step_number == 3
        assert verifier_entries[0].step_number == 7
