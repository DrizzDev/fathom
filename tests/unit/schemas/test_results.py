from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.artifacts import ScreenArtifact
from fathom.schemas.results import (
    ActionTraceAttempt,
    ActionTraceEvent,
    TraceEmission,
)
from fathom.schemas.screens import ScreenCapture


class TestTraceEmission(unittest.TestCase):
    """
    Pin the TraceEmission adapter envelope; composes an ActionTraceEvent with an optional staged artifact handle.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Build a minimal ScreenCapture fixture used by every emission test in this class.
        """

        return ScreenCapture(
            width=1080,
            height=2400,
            activity="com.test.app",
            image=b"png",
            timestamp=1_714_200_000_000,
        )

    def test_emission_defaults_artifact_to_none(self) -> None:
        """
        An emission wraps only the event when the pipeline did not produce an artifact handle.
        """

        event = ActionTraceEvent(capture=self.__capture(), coords=(10, 20))
        emission = TraceEmission(event=event)

        self.assertIs(emission.event, event)
        self.assertIsNone(emission.artifact)

    def test_emission_carries_artifact_when_pipeline_staged(self) -> None:
        """
        A successful pipeline staging surfaces a ScreenArtifact handle alongside the source event.
        """

        event = ActionTraceEvent(
            capture=self.__capture(),
            coords=(15, 25),
            attempt=ActionTraceAttempt(index=2),
        )
        artifact = ScreenArtifact(uri="cdn://trace-2", width=1080, height=2400)

        emission = TraceEmission(event=event, artifact=artifact)

        self.assertIs(emission.event, event)
        self.assertIsNotNone(emission.artifact)
        assert emission.artifact is not None
        self.assertEqual(emission.artifact.uri, "cdn://trace-2")

    def test_emission_is_frozen(self) -> None:
        """
        TraceEmission is immutable to prevent downstream consumers from mutating the adapter outcome in place.
        """

        emission = TraceEmission(event=ActionTraceEvent(capture=self.__capture(), coords=(0, 0)))

        with self.assertRaises(ValidationError):
            emission.artifact = ScreenArtifact(uri="cdn://x")  # type: ignore[misc]
