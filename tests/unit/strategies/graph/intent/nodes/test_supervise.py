from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.strategies.graph.intent.nodes.supervise import SuperviseNode


class SuperviseNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the SUPERVISE node's cancellation and missing-state branches.

    SUPERVISE runs localization, supervision, and bounded healing against
    the planned step. Both pins verify that the node degrades gracefully
    when the prerequisite state is absent: cancellation propagates the
    cancellation reason; missing planned step or capture emits a
    ``SHOULD_RETRY`` signal so the router re-enters GROUND instead of
    letting the silent EXECUTE→OBSERVE→RECORD cascade fail downstream
    with a misleading ``record.missing.step_result`` Sentry alert.
    """

    @staticmethod
    def __provider(*, cancelled: bool = False) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the cancellation
        check, workflow id, and persistence helper used by the early
        branches. The gate / observer / localizer surfaces stay unmocked
        because they must not be reached on these branches.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.persistence.persist = MagicMock()
        return provider

    async def test_cancellation_marks_complete(self) -> None:
        """
        A cancelled run must terminate with :attr:`CompletionReason.CANCELLED`.
        """

        provider = self.__provider(cancelled=True)
        node = SuperviseNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )

    async def test_missing_planned_step_or_capture_signals_should_retry(self) -> None:
        """
        Missing planned step or capture must publish ``SHOULD_RETRY`` so
        :meth:`IntentGraphBuilder.__route_after_supervise` routes back to
        GROUND. Without the signal the router would fall through to
        EXECUTE on a partial state, which is the cascade that surfaced
        on staging as the ``record.missing.step_result`` Sentry alert.
        """

        provider = self.__provider(cancelled=False)
        node = SuperviseNode(provider=provider)

        result: Any = await node(
            state={  # type: ignore[arg-type]
                CommonStateKey.CAPTURE: None,
                IntentStateKey.PLANNED_STEP: None,
            },
        )

        self.assertEqual(result, {IntentStateKey.SHOULD_RETRY: True})
        provider.persistence.persist.assert_called_once_with(
            result={IntentStateKey.SHOULD_RETRY: True},
        )
