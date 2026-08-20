from __future__ import annotations

import unittest
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from fathom.runtime.temporal.activities import FathomActivities
from fathom.settings.env import FathomSettings


class _RecordingAdapter:
    """
    Minimal stand-in for an adapter that supports cleanup().

    Records the order of cleanup calls into a shared list so the test can
    verify partial-build drains every adapter in reverse-creation order.
    """

    def __init__(self, *, name: str, call_log: list[str]) -> None:
        """
        Initialize with a unique name and a shared call-log list.
        """

        self.name = name
        self.__call_log = call_log
        self.cleanup = AsyncMock(side_effect=self.__record)

    async def __record(self) -> None:
        """
        Append this adapter's name to the shared call-log so the test can
        assert reverse-creation drain order.
        """

        self.__call_log.append(self.name)


class _RecordingCloseOnlyAdapter:
    """
    Adapter that exposes close() (sync) instead of cleanup() — exercises the
    fallback branch of the drain code.
    """

    def __init__(self, *, name: str, call_log: list[str]) -> None:
        """
        Initialize with a unique name and a shared call-log list.
        """

        self.name = name
        self.__call_log = call_log

    def close(self) -> None:
        """
        Sync close; records the call into the shared log.
        """

        self.__call_log.append(self.name)


class _RecordingSyncCleanupAdapter:
    """
    Adapter that exposes a synchronous cleanup() method.
    """

    def __init__(self, *, name: str, call_log: list[str]) -> None:
        """
        Initialize with a unique name and a shared call-log list.
        """

        self.name = name
        self.__call_log = call_log

    def cleanup(self) -> None:
        """
        Sync cleanup; records the call into the shared log.
        """

        self.__call_log.append(self.name)


class _RecordingInteractionAdapter:
    """
    Adapter shaped like InteractionPort: aclose() is teardown, while cleanup()
    is a domain method that requires a request argument.
    """

    def __init__(self, *, name: str, call_log: list[str]) -> None:
        """
        Initialize with a unique name and a shared call-log list.
        """

        self.name = name
        self.__call_log = call_log
        self.initialized = False
        self.initialize = AsyncMock(side_effect=self.__initialize)
        self.aclose = AsyncMock(side_effect=self.__record)

    async def __initialize(self) -> None:
        """
        Mark the adapter as initialized.
        """

        self.initialized = True

    async def __record(self) -> None:
        """
        Append this adapter's name when the teardown path is called.
        """

        self.__call_log.append(self.name)

    async def cleanup(self, *, request: object) -> None:
        """
        Domain cleanup operation; must not be called by partial-build teardown.
        """

        _ = request
        raise AssertionError("Domain cleanup must not be used as teardown")


class _InitializedInteractionBuilder:
    """
    Fluent builder double that rejects uninitialized interaction adapters.
    """

    def __getattr__(self, name: str) -> object:
        """
        Return fluent no-op methods for builder steps irrelevant to this test.
        """

        if name == "build":
            return self.__build

        return self.__fluent

    def with_interaction(
        self, *, port: _RecordingInteractionAdapter
    ) -> "_InitializedInteractionBuilder":
        """
        Accept only an adapter that has already completed initialize().
        """

        if not port.initialized:
            raise AssertionError("Interaction adapter reached builder before initialize()")

        return self

    def __fluent(self, **_: object) -> "_InitializedInteractionBuilder":
        """
        Preserve the runtime builder's fluent call style.
        """

        return self

    def __build(self) -> object:
        """
        Return a runner placeholder if the test reaches build().
        """

        return object()


class FathomActivitiesPartialBuildTest(unittest.IsolatedAsyncioTestCase):
    """
    If any step after partial adapters are created fails during __build_runner, every registered
    adapter must be drained — telemetry, storage, device, and perception, not just the LLMs.
    """

    async def test_partial_build_drains_every_registered_adapter(self) -> None:
        """
        Simulate composer.compose() failing after every adapter has been
        constructed. Every adapter that was registered must have its cleanup
        path invoked exactly once, in reverse-creation order.
        """

        call_log: List[str] = []
        activities = FathomActivities(settings=FathomSettings())

        planner_llm = _RecordingAdapter(name="planner", call_log=call_log)
        device_adapter = _RecordingAdapter(name="device", call_log=call_log)
        perception_adapter = _RecordingAdapter(name="perception", call_log=call_log)

        signal_adapter = _RecordingAdapter(name="signal", call_log=call_log)
        storage_adapter = _RecordingAdapter(name="storage", call_log=call_log)
        telemetry_adapter = _RecordingCloseOnlyAdapter(name="telemetry", call_log=call_log)

        interaction_adapter = _RecordingInteractionAdapter(name="interaction", call_log=call_log)

        request = MagicMock()
        request.objective = MagicMock(use_xml=False)
        request.interaction = MagicMock()
        request.interaction.qualifier_configuration = MagicMock()
        request.interaction.realignment = None
        request.interaction.intent_configuration = MagicMock()
        request.interaction.execution_configuration = MagicMock()
        request.interaction.exploration_configuration = MagicMock()
        request.runtime = MagicMock(interactive=False)

        with (
            patch("fathom.runtime.temporal.activities.LLMFactory") as llm_factory_cls,
            patch("fathom.runtime.temporal.activities.DeviceFactory") as device_factory_cls,
            patch("fathom.runtime.temporal.activities.PerceptionFactory") as perception_factory_cls,
            patch("fathom.runtime.temporal.activities.TelemetryFactory") as telemetry_factory_cls,
            patch("fathom.runtime.temporal.activities.StorageFactory") as storage_factory_cls,
            patch("fathom.runtime.temporal.activities.SignalFactory") as signal_factory_cls,
            patch(
                "fathom.runtime.temporal.activities.InteractionFactory"
            ) as interaction_factory_cls,
            patch("fathom.runtime.temporal.activities.Fathom") as fathom_cls,
            patch("fathom.runtime.temporal.activities.QualifierComposer") as composer_cls,
            patch.object(
                activities,
                "_FathomActivities__assembly",
                MagicMock(
                    build_device_configuration=MagicMock(return_value=MagicMock()),
                    build_planner_model_configuration=MagicMock(return_value=MagicMock()),
                    build_storage_configuration=MagicMock(return_value=MagicMock()),
                    build_telemetry_configuration=MagicMock(return_value=MagicMock()),
                ),
            ),
        ):
            device_factory_cls.return_value.create = MagicMock(return_value=device_adapter)
            perception_factory_cls.return_value.create = MagicMock(return_value=perception_adapter)

            llm_factory_instance = MagicMock()
            llm_factory_instance.create = MagicMock(return_value=planner_llm)
            llm_factory_cls.return_value = llm_factory_instance

            telemetry_factory_cls.return_value.create = MagicMock(return_value=telemetry_adapter)
            storage_factory_cls.return_value.create = MagicMock(return_value=storage_adapter)
            signal_factory_cls.return_value.create = MagicMock(return_value=signal_adapter)
            interaction_factory_cls.return_value.create = MagicMock(
                return_value=interaction_adapter
            )
            fathom_cls.builder = MagicMock(return_value=_InitializedInteractionBuilder())

            composer_cls.should_compose = MagicMock(return_value=True)
            composer_cls.return_value.compose = MagicMock(
                side_effect=RuntimeError("simulated build failure after all adapters built")
            )

            with self.assertRaises(RuntimeError):
                await activities._FathomActivities__build_runner(  # type: ignore[attr-defined]
                    workflow_id="wf-test",
                    request=request,
                )

        # Every registered adapter drained exactly once, in reverse-creation order.
        interaction_adapter.initialize.assert_awaited_once_with()
        self.assertEqual(
            call_log,
            ["interaction", "storage", "telemetry", "planner", "perception", "device", "signal"],
        )

    async def test_partial_build_isolates_per_resource_errors(self) -> None:
        """
        A failing cleanup on one adapter must not prevent the others from
        draining. Verifies the per-resource try/except in __drain_partial_resources.
        """

        activities = FathomActivities(settings=FathomSettings())
        call_log: List[str] = []

        good_one = _RecordingAdapter(name="good_one", call_log=call_log)
        bad = _RecordingAdapter(name="bad", call_log=call_log)
        bad.cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        good_two = _RecordingAdapter(name="good_two", call_log=call_log)

        await activities._FathomActivities__drain_partial_resources(  # type: ignore[attr-defined]
            resources=[good_one, bad, good_two]
        )

        # good_two drains first (reverse order); bad raises but doesn't skip good_one.
        self.assertIn("good_one", call_log)
        self.assertIn("good_two", call_log)
        self.assertEqual(call_log, ["good_two", "good_one"])

    async def test_partial_build_skips_adapters_without_cleanup_or_close(self) -> None:
        """
        Adapters with neither cleanup() nor close() must be skipped silently;
        they have no teardown contract and probing must not raise.
        """

        activities = FathomActivities(settings=FathomSettings())

        class _Inert:
            """Adapter with no teardown methods at all."""

        # Should not raise even with no teardown methods present.
        await activities._FathomActivities__drain_partial_resources(  # type: ignore[attr-defined]
            resources=[_Inert(), _Inert()]
        )

    async def test_partial_build_supports_sync_cleanup(self) -> None:
        """
        Future adapters may expose sync cleanup(); the drain path should handle
        them the same way it already handles sync close().
        """

        activities = FathomActivities(settings=FathomSettings())
        call_log: List[str] = []
        adapter = _RecordingSyncCleanupAdapter(name="sync-cleanup", call_log=call_log)

        await activities._FathomActivities__drain_partial_resources(  # type: ignore[attr-defined]
            resources=[adapter]
        )

        self.assertEqual(call_log, ["sync-cleanup"])

    async def test_partial_build_prefers_aclose_over_domain_cleanup(self) -> None:
        """
        InteractionPort exposes cleanup(request=...) as a domain operation.
        Partial-build teardown must call aclose() instead.
        """

        call_log: List[str] = []
        activities = FathomActivities(settings=FathomSettings())
        adapter = _RecordingInteractionAdapter(name="interaction", call_log=call_log)

        await activities._FathomActivities__drain_partial_resources(  # type: ignore[attr-defined]
            resources=[adapter]
        )

        adapter.aclose.assert_awaited_once_with()
        self.assertEqual(call_log, ["interaction"])


_ = Any  # exported for downstream test extensions; binding silences unused warning


if __name__ == "__main__":
    unittest.main()
