from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from fathom.interfaces.llm import LLMPort
from fathom.runtime.temporal.activities import FathomActivities
from fathom.schemas.composition import RunnerComposition


class FathomActivitiesTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover cleanup ordering in Temporal activities.
    """

    async def test_execute_intent_cancels_runner_before_cleanup(self) -> None:
        """
        Signal cancellation before tearing down runner resources, including
        any infrastructure the composition root tracks via RunnerComposition.
        """

        activities = FathomActivities(settings=Mock())
        cleanup_order: list[str] = []

        runner = Mock()
        runner.device = Mock()
        runner.device.get_current_package = AsyncMock(return_value="com.example.delivery")
        runner.run_intent = AsyncMock(side_effect=RuntimeError("boom"))
        runner.cancel = Mock(side_effect=lambda: cleanup_order.append("cancel"))
        runner.cleanup = AsyncMock(side_effect=lambda: cleanup_order.append("cleanup"))
        runner.notify = AsyncMock()

        owned_llm = MagicMock(spec=LLMPort)
        owned_llm.cleanup = AsyncMock(side_effect=lambda: cleanup_order.append("owned-llm"))

        composition = RunnerComposition(runner=runner, resources=(owned_llm,))

        validated_request = SimpleNamespace(
            objective=SimpleNamespace(
                intent="Open app",
                max_steps=1,
                package_name=None,
                use_xml=True,
            ),
            runtime=SimpleNamespace(interactive=True),
            memory=SimpleNamespace(context_scope="execution", conversation_id="conversation"),
            interaction=SimpleNamespace(realignment=None),
        )

        release = Mock()

        with (
            patch.object(
                activities,
                "_FathomActivities__validate_intent_request",
                return_value=validated_request,
            ),
            patch.object(
                activities,
                "_FathomActivities__build_runner",
                new=AsyncMock(return_value=composition),
            ),
            patch(
                "fathom.runtime.temporal.activities.SignalStateRegistry.shared",
                return_value=SimpleNamespace(release=release),
            ),
        ):
            result = await activities.execute_intent("workflow-id", {})

        self.assertEqual(cleanup_order, ["cancel", "cleanup", "owned-llm"])
        self.assertFalse(result["success"])
        runner.cleanup.assert_awaited_once_with()
        owned_llm.cleanup.assert_awaited_once_with()
        runner.notify.assert_not_awaited()
        release.assert_called_once_with(workflow_id="workflow-id")

    async def test_execute_intent_releases_signal_state_when_build_fails(self) -> None:
        """
        Build failures happen before a RunnerComposition exists. Temporal signal
        state must still be released so failed activities cannot accumulate
        process-scoped workflow state.
        """

        activities = FathomActivities(settings=Mock())
        release = Mock()
        validated_request = SimpleNamespace(
            objective=SimpleNamespace(intent="Open app", max_steps=1),
            runtime=SimpleNamespace(interactive=False),
        )

        with (
            patch.object(
                activities,
                "_FathomActivities__validate_intent_request",
                return_value=validated_request,
            ),
            patch.object(
                activities,
                "_FathomActivities__build_runner",
                new=AsyncMock(side_effect=RuntimeError("build failed")),
            ),
            patch(
                "fathom.runtime.temporal.activities.SignalStateRegistry.shared",
                return_value=SimpleNamespace(release=release),
            ),
            self.assertRaises(RuntimeError),
        ):
            await activities.execute_intent("workflow-id", {})

        release.assert_called_once_with(workflow_id="workflow-id")

    async def test_execute_exploration_releases_signal_state_when_build_fails(self) -> None:
        """
        Exploration uses the same Temporal composition path; failed runner builds
        must also release signal state there.
        """

        activities = FathomActivities(settings=Mock())
        release = Mock()
        validated_request = SimpleNamespace(
            objective=SimpleNamespace(max_steps=1),
            runtime=SimpleNamespace(interactive=False),
        )

        with (
            patch.object(
                activities,
                "_FathomActivities__validate_exploration_request",
                return_value=validated_request,
            ),
            patch.object(
                activities,
                "_FathomActivities__build_runner",
                new=AsyncMock(side_effect=RuntimeError("build failed")),
            ),
            patch(
                "fathom.runtime.temporal.activities.SignalStateRegistry.shared",
                return_value=SimpleNamespace(release=release),
            ),
            self.assertRaises(RuntimeError),
        ):
            await activities.execute_exploration("workflow-id", {})

        release.assert_called_once_with(workflow_id="workflow-id")
