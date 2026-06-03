from __future__ import annotations

import asyncio
import time
import unittest
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from fathom.constants import ActionType
from fathom.constants.reasoning import USER_GUIDANCE_ANALYZE_TTL
from fathom.core.context.manager import ContextManager
from fathom.schemas.actions import Action
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
        assert context["guidance"] == [
            f"[active, remaining_analyze_turns={USER_GUIDANCE_ANALYZE_TTL}] use the search bar"
        ]
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

    @pytest.mark.asyncio
    async def test_user_guidance_survives_one_analyze_then_expires_after_ttl(
        self, manager: ContextManager
    ) -> None:
        """
        HITL guidance must not disappear after one ignored planner turn,
        but it must expire before becoming stale prompt pressure.
        """

        await manager.inject_user_guidance(guidance="tap Continue", step=4)

        manager.consume_user_guidance()
        entries = manager.get_user_guidance()
        assert len(entries) == 1
        assert entries[0].remaining_analyses == USER_GUIDANCE_ANALYZE_TTL - 1

        for _ in range(USER_GUIDANCE_ANALYZE_TTL - 1):
            manager.consume_user_guidance()

        assert manager.get_user_guidance() == []
        assert manager.get_full_context()["guidance"] == []

    @pytest.mark.asyncio
    async def test_multiple_user_guidance_entries_preserve_order(
        self, manager: ContextManager
    ) -> None:
        """
        Multiple HITL instructions should remain independent and ordered.
        """

        await manager.inject_user_guidance(guidance="use test login", step=1)
        await manager.inject_user_guidance(guidance="skip marketing permissions", step=2)

        assert [entry.content for entry in manager.get_user_guidance()] == [
            "use test login",
            "skip marketing permissions",
        ]


class ContextManagerShutdownTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression for the 30-second cleanup hang observed on production intent runs.

    The previous design spawned a persistence worker at construction time and
    registered it into __background_tasks. shutdown() then awaited every task
    in that set with a 30 second drain timeout. The persistence worker blocks
    forever on `await self.__persist_queue.get()` (the queue is never written
    to because __enqueue_persist is a no-op), so every successful run paid a
    structural 30 second wait at cleanup.

    The fix disables both the spawn and the registration. shutdown() must now
    return promptly because __background_tasks stays empty for the lifetime of
    a single-step run.
    """

    async def test_construction_does_not_spawn_persistence_task(self) -> None:
        """
        Construction must not start the persistence worker. The persistence path
        is disabled; spawning it pollutes __background_tasks and blocks shutdown.
        """

        manager = ContextManager(memory=MagicMock())

        self.assertIsNone(manager._ContextManager__persistence_task)  # type: ignore[attr-defined]
        self.assertEqual(manager._ContextManager__background_tasks, set())  # type: ignore[attr-defined]

    async def test_shutdown_returns_promptly_when_no_background_work(self) -> None:
        """
        A run with no summarization (single-step, branch() not triggered) must
        shutdown in under one second — far below the 30s drain timeout that
        prod was paying on every successful run.
        """

        manager = ContextManager(memory=MagicMock())

        started = time.perf_counter()
        await manager.shutdown()
        elapsed = time.perf_counter() - started

        self.assertLess(
            elapsed,
            1.0,
            msg=(
                "shutdown must not wait on the disabled persistence worker. "
                f"Took {elapsed:.2f}s; regression — re-check whether "
                "__start_persistence_loop() or its background_tasks.add() were "
                "re-enabled without also disabling the drain logic."
            ),
        )

    async def test_commit_does_not_grow_persist_queue(self) -> None:
        """
        Regression for the unbounded-queue leak in __enqueue_persist.

        Every step of every workflow calls commit(); commit() calls
        __enqueue_persist(). If __enqueue_persist still pushes to
        __persist_queue while the persistence worker is disabled, the queue
        grows by one entry per step for the worker process lifetime — a real
        memory leak in long-running Temporal workers.

        Per the disabled-persistence design, __enqueue_persist must return
        early without touching the queue.
        """

        manager = ContextManager(memory=MagicMock())

        action = Action(action_type=ActionType.TAP, rationale="test")

        for _ in range(50):
            await manager.commit(observation="ob", thought="th", action=action)

        self.assertEqual(
            manager._ContextManager__persist_queue.qsize(),  # type: ignore[attr-defined]
            0,
            msg=(
                "__enqueue_persist must early-return while persistence is "
                "disabled — every queue.put_nowait that lands here is a leaked "
                "snapshot the disabled worker will never drain."
            ),
        )

    async def test_shutdown_drains_real_background_summarization_tasks(self) -> None:
        """
        When __background_tasks DOES contain a real summarization task, shutdown
        must still drain it (or cancel it on timeout). Proves the fix to the
        persistence loop didn't accidentally bypass the legitimate drain path.
        """

        manager = ContextManager(memory=MagicMock())

        completed_marker: dict[str, bool] = {"done": False}

        async def fake_summarization_task() -> None:
            """
            Stand-in summarization that finishes well under the drain timeout.
            """

            await asyncio.sleep(0.05)
            completed_marker["done"] = True

        task = asyncio.create_task(fake_summarization_task())
        manager._ContextManager__background_tasks.add(task)  # type: ignore[attr-defined]
        task.add_done_callback(
            manager._ContextManager__background_tasks.discard  # type: ignore[attr-defined]
        )

        await manager.shutdown()

        self.assertTrue(completed_marker["done"])


if __name__ == "__main__":
    unittest.main()
