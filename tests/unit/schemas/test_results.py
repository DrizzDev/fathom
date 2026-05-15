import unittest

from fathom.schemas.artifacts import ScreenArtifact
from fathom.schemas.results import ExecutionResult


class TestExecutionResultTraceArtifact(unittest.TestCase):
    """The ``trace_artifact`` field carries the annotated pre-action image URI
    from ``ActionExecutor`` back to the intent node so it can be surfaced in
    ``StepArtifacts.screen.trace``.
    """

    def test_default_is_none(self) -> None:
        result = ExecutionResult(success=False, duration=0)
        self.assertIsNone(result.trace_artifact)

    def test_carries_screen_artifact(self) -> None:
        artifact = ScreenArtifact(uri="gs://bucket/traces/step_1_tap.png")

        result = ExecutionResult(success=True, duration=42, trace_artifact=artifact)

        self.assertIsNotNone(result.trace_artifact)
        self.assertEqual(result.trace_artifact.uri, "gs://bucket/traces/step_1_tap.png")

    def test_serialization_includes_trace_artifact(self) -> None:
        result = ExecutionResult(
            success=True,
            duration=42,
            trace_artifact=ScreenArtifact(uri="gs://x/y.png"),
        )

        payload = result.model_dump(mode="json")

        self.assertEqual(payload["trace_artifact"]["uri"], "gs://x/y.png")

    def test_model_copy_attaches_trace_artifact(self) -> None:
        """ActionExecutor.act() uses ``model_copy(update={...})`` to attach
        the trace because ExecutionResult is frozen. Verify that pattern works."""
        base = ExecutionResult(success=True, duration=42)
        self.assertIsNone(base.trace_artifact)

        with_trace = base.model_copy(
            update={"trace_artifact": ScreenArtifact(uri="gs://x/y.png")}
        )

        self.assertIsNotNone(with_trace.trace_artifact)
        self.assertEqual(with_trace.trace_artifact.uri, "gs://x/y.png")
        # Base remains unchanged (frozen, immutable copies)
        self.assertIsNone(base.trace_artifact)


if __name__ == "__main__":
    unittest.main()
