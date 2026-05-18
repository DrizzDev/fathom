import unittest

from fathom.constants import ActionType
from fathom.constants.storage import StorageBackend
from fathom.schemas.actions import Action
from fathom.schemas.artifacts import (
    ScreenArtifact,
    ScreenArtifactBundle,
    StepArtifacts,
)
from fathom.schemas.steps import Step, StepResult


class TestStepResultArtifacts(unittest.TestCase):
    """Unit tests covering the optional `artifacts` field on `StepResult`."""

    @staticmethod
    def __build_step(*, step_number: int = 0) -> Step:
        """Build a minimal `Step` instance suitable for `StepResult` construction."""

        return Step(
            action=Action(
                target="element",
                action_type=ActionType.TAP,
                rationale="unit-test action",
            ),
            step_number=step_number,
            screen_hash="0123456789abcdef",
        )

    def test_step_result_defaults_artifacts_to_none(self) -> None:
        """Existing call sites that omit `artifacts` continue to work."""

        step_result = StepResult(
            success=True,
            duration=120,
            screen_changed=True,
            step=self.__build_step(),
            pre_hash="aaaaaaaaaaaaaaaa",
            post_hash="bbbbbbbbbbbbbbbb",
        )

        self.assertIsNone(step_result.artifacts)

    def test_step_result_accepts_screen_before_and_after(self) -> None:
        """Artifacts surface as a nested `screen.before` / `screen.after` payload."""

        artifacts = StepArtifacts(
            screen=ScreenArtifactBundle(
                before=ScreenArtifact(
                    visual_hash="aaaa",
                    uri="/tmp/before.png",
                    storage_backend=StorageBackend.LOCAL,
                ),
                after=ScreenArtifact(
                    visual_hash="bbbb",
                    uri="/tmp/after.png",
                    storage_backend=StorageBackend.LOCAL,
                ),
            ),
        )

        step_result = StepResult(
            step=self.__build_step(),
            success=True,
            duration=120,
            pre_hash="aaaa",
            post_hash="bbbb",
            screen_changed=True,
            artifacts=artifacts,
        )

        self.assertIsNotNone(step_result.artifacts)
        assert step_result.artifacts is not None  # narrow for type-checker

        self.assertIsNotNone(step_result.artifacts.screen)
        assert step_result.artifacts.screen is not None

        self.assertEqual(step_result.artifacts.screen.before.uri, "/tmp/before.png")
        self.assertEqual(step_result.artifacts.screen.after.uri, "/tmp/after.png")

    def test_step_result_serializes_artifacts_namespaced(self) -> None:
        """JSON serialization preserves the nested artifact namespace consumed by telemetry adapters."""

        step_result = StepResult(
            duration=80,
            success=True,
            pre_hash="aaaa",
            post_hash="bbbb",
            step=self.__build_step(),
            screen_changed=False,
            artifacts=StepArtifacts(
                screen=ScreenArtifactBundle(
                    before=ScreenArtifact(uri="/tmp/before.png"),
                ),
            ),
        )

        payload = step_result.model_dump(mode="json")

        self.assertIn("artifacts", payload)
        self.assertIn("screen", payload["artifacts"])
        self.assertEqual(payload["artifacts"]["screen"]["before"]["uri"], "/tmp/before.png")
        self.assertIsNone(payload["artifacts"]["screen"]["after"])

    def test_step_result_to_record_does_not_require_artifacts(self) -> None:
        """`to_record()` produces the same persistence shape regardless of artifact presence."""

        without_artifacts = StepResult(
            duration=50,
            success=True,
            pre_hash="aaaa",
            post_hash="bbbb",
            screen_changed=False,
            step=self.__build_step(step_number=0),
        )
        with_artifacts = StepResult(
            duration=50,
            success=True,
            pre_hash="aaaa",
            post_hash="bbbb",
            screen_changed=False,
            step=self.__build_step(step_number=0),
            artifacts=StepArtifacts(
                screen=ScreenArtifactBundle(
                    before=ScreenArtifact(uri="/tmp/before.png"),
                ),
            ),
        )

        self.assertEqual(
            without_artifacts.to_record().model_dump(mode="json"),
            with_artifacts.to_record().model_dump(mode="json"),
        )
