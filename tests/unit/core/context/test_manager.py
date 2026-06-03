from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import MagicMock

from fathom.constants import ActionType
from fathom.core.context.manager import ContextManager
from fathom.schemas.actions import Action


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
