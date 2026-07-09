import unittest

from fathom.constants import ActionType
from fathom.constants.storage import StorageBackend
from fathom.schemas.actions import Action
from fathom.schemas.artifacts import (
    ScreenArtifact,
    ScreenArtifactBundle,
    StepArtifacts,
)
from fathom.schemas.capture import Capture, CaptureRequest
from fathom.schemas.steps import Step, StepGoal, StepHistory, StepResult


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
        screen = step_result.artifacts.screen
        assert screen is not None
        assert screen.before is not None
        assert screen.after is not None

        self.assertEqual(screen.before.uri, "/tmp/before.png")
        self.assertEqual(screen.after.uri, "/tmp/after.png")

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

    def test_step_result_to_record_carries_artifacts_when_present(self) -> None:
        """`to_record()` persists artifacts when they are present."""

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

        self.assertIsNone(without_artifacts.to_record().artifacts)
        self.assertEqual(with_artifacts.to_record().artifacts, with_artifacts.artifacts)

    def test_step_result_to_record_carries_goal_context(self) -> None:
        """`to_record()` persists compact sub-goal context when supplied."""

        goal = StepGoal(index=2, description="Check rating", directive="validate")
        record = StepResult(
            duration=50,
            success=True,
            pre_hash="aaaa",
            post_hash="bbbb",
            screen_changed=False,
            step=self.__build_step(step_number=5),
        ).to_record(goal=goal)

        self.assertIsNotNone(record.goal)
        assert record.goal is not None
        self.assertEqual(record.goal.index, 2)
        self.assertEqual(record.goal.description, "Check rating")
        self.assertEqual(record.goal.directive, "validate")


class TestStepResultCapture(unittest.TestCase):
    """Covers persistence of STORE capture metadata onto the step record and history."""

    @staticmethod
    def __step(*, action_type: ActionType) -> Step:
        """Build a minimal Step for the given action type."""

        return Step(
            action=Action(action_type=action_type, target="element", rationale="r"),
            step_number=3,
            screen_hash="0123456789abcdef",
        )

    def __result(self, *, action_type: ActionType, capture: object) -> StepResult:
        """Build a step result carrying the given capture."""

        return StepResult(
            step=self.__step(action_type=action_type),
            success=True,
            duration=10,
            screen_changed=False,
            pre_hash="aaaa",
            post_hash="bbbb",
            capture=capture,  # type: ignore[arg-type]
        )

    def test_store_success_record_has_capture(self) -> None:
        """A successful STORE persists its capture (value, success) onto the record."""

        record = self.__result(
            action_type=ActionType.STORE,
            capture=Capture.succeeded(name="abc", value="₹499", step=3),
        ).to_record()

        self.assertIsNotNone(record.capture)
        assert record.capture is not None
        self.assertTrue(record.capture.success)
        self.assertEqual(record.capture.value, "₹499")

    def test_store_failed_record_has_capture(self) -> None:
        """A failed STORE persists its failed capture (success=False, reason) onto the record."""

        record = self.__result(
            action_type=ActionType.STORE,
            capture=Capture.failed(name="abc", reason="no element", step=3),
        ).to_record()

        self.assertIsNotNone(record.capture)
        assert record.capture is not None
        self.assertFalse(record.capture.success)
        self.assertEqual(record.capture.reason, "no element")

    def test_non_store_record_has_no_capture(self) -> None:
        """A non-STORE step persists no capture."""

        record = self.__result(action_type=ActionType.TAP, capture=None).to_record()

        self.assertIsNone(record.capture)

    def test_old_history_without_capture_still_loads(self) -> None:
        """A history record predating the capture field loads with capture defaulting to None."""

        history = StepHistory.model_validate(
            {
                "history": [
                    {
                        "step_number": 0,
                        "action_type": "tap",
                        "target": "Login button",
                        "success": True,
                        "screen_changed": True,
                        "duration": 12,
                    }
                ]
            }
        )

        self.assertEqual(len(history.history), 1)
        self.assertIsNone(history.history[0].capture)

    def test_history_export_includes_capture(self) -> None:
        """The serialized record exposes the capture payload for downstream script generation."""

        payload = (
            self.__result(
                action_type=ActionType.STORE,
                capture=Capture.succeeded(name="abc", value="₹499", step=3),
            )
            .to_record()
            .model_dump(mode="json")
        )

        self.assertIn("capture", payload)
        self.assertIsNotNone(payload["capture"])
        self.assertEqual(payload["capture"]["value"], "₹499")

    def test_store_record_carries_capture_request(self) -> None:
        """A STORE step persists the request (name/subject/value) that drives script generation."""

        action = Action(
            action_type=ActionType.STORE,
            target="element",
            rationale="r",
            capture=CaptureRequest(name="abc", subject="price", value="₹499"),
        )
        result = StepResult(
            step=Step(action=action, step_number=3, screen_hash="0123456789abcdef"),
            success=True,
            duration=10,
            screen_changed=False,
            pre_hash="aaaa",
            post_hash="bbbb",
            capture=Capture.succeeded(name="abc", value="₹499", step=3),
        )

        record = result.to_record()

        self.assertIsNotNone(record.capture_request)
        assert record.capture_request is not None
        self.assertEqual(record.capture_request.subject, "price")
        self.assertEqual(record.capture_request.value, "₹499")

    def test_non_store_record_has_no_capture_request(self) -> None:
        """A non-STORE step persists no capture request."""

        record = self.__result(action_type=ActionType.TAP, capture=None).to_record()

        self.assertIsNone(record.capture_request)
