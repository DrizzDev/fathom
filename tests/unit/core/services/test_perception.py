from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Optional
from unittest.mock import Mock

from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.artifact.renderer import PassthroughRenderer
from fathom.core.services.perception import PerceptionService
from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.storage import StoragePort
from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactMetadata,
    ArtifactReceipt,
    PipelineConfiguration,
)
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class FakeStorage(StoragePort):
    """
    In-memory :class:`StoragePort` double that records every save call
    and returns a configurable storage identifier.
    """

    def __init__(self, *, storage_id: str = "/tmp/fathom/test/screenshot.png") -> None:
        """
        Configure the identifier returned by save() on every call.
        """

        self.__storage_id = storage_id
        self.calls: List[Dict[str, Any]] = []

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record the save invocation and return the configured identifier.
        """

        self.calls.append({"data_len": len(data), "metadata": dict(metadata or {})})
        return self.__storage_id


class FakePerception(PerceptionPort):
    """
    PerceptionPort double that returns a pre-built capture and counts capture() invocations.
    """

    def __init__(self, *, capture: ScreenCapture) -> None:
        """
        Bind the double to the supplied capture instance.
        """

        self.__capture = capture
        self.call_count = 0

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return the device runtime configuration; unused by these tests.
        """

        return None

    async def capture(self) -> ScreenCapture:
        """
        Return the pre-built capture and increment the invocation count.
        """

        self.call_count += 1
        return self.__capture


def _build_capture() -> ScreenCapture:
    return ScreenCapture(
        width=1080,
        height=2400,
        activity="com.test.app",
        image=b"fake-png-bytes",
        timestamp=1714200000000,
        metadata={},
    )


class TestPerceptionServiceFallbackBranch(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the pipeline-less ``perceive`` path that persists via
    :class:`StoragePort` directly.
    """

    async def test_perceive_persists_screenshot_with_pre_action_phase(self) -> None:
        """
        Pre-action saves carry ``phase=pre_action`` and identity metadata.
        """

        storage = FakeStorage()
        perception = FakePerception(capture=_build_capture())
        service = PerceptionService(
            storage=storage,
            perception=perception,
            hierarchy_signature_builder=Mock(),
        )

        await service.perceive(session_id="session-1", step_number=1)

        self.assertEqual(len(storage.calls), 1)
        metadata = storage.calls[0]["metadata"]
        self.assertEqual(metadata["type"], "screenshot")
        self.assertEqual(metadata["phase"], "pre_action")
        self.assertEqual(metadata["session_id"], "session-1")
        self.assertEqual(metadata["package_name"], "com.test.app")
        self.assertEqual(metadata["activity_name"], "com.test.app")
        self.assertIn("timestamp", metadata)

    async def test_perceive_stamps_storage_id_onto_metadata_and_screenshot_uri(self) -> None:
        """
        The fallback path exposes the storage identifier through both the
        legacy ``metadata['storage_id']`` key and the typed
        ``screenshot_uri`` field so downstream consumers can read either.
        """

        storage = FakeStorage(storage_id="storage://artifact-id")
        perception = FakePerception(capture=_build_capture())
        service = PerceptionService(
            storage=storage,
            perception=perception,
            hierarchy_signature_builder=Mock(),
        )

        result = await service.perceive(session_id="session-1", step_number=1)

        self.assertEqual(result.metadata["storage_id"], "storage://artifact-id")
        self.assertEqual(result.screenshot_uri, "storage://artifact-id")
        self.assertNotIn("path", result.metadata)


class _RecordingSink(ArtifactSinkPort):
    """
    :class:`ArtifactSinkPort` double that captures every persist call
    without touching disk so tests can assert which artifact kinds the
    pipeline drove through.
    """

    def __init__(self) -> None:
        """
        Initialise the record of persisted metadata observations.
        """

        self.persisted: List[ArtifactMetadata] = []

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
    ) -> ArtifactReceipt:
        """
        Record the persist invocation and return a stable receipt.
        """

        self.persisted.append(metadata)
        return ArtifactReceipt(identifier="cloud://artifact", local_cleanup=True)


class _TempPathManager:
    """
    Minimal path manager double stubbed against a temp directory root,
    exposing only the surface :class:`PerceptionService` exercises.
    """

    def __init__(self, *, root: Path) -> None:
        """
        Bind the manager to the supplied temp directory root.
        """

        self.__root = root

    def get_xml_path(self, package_name: str, session_id: str, filename: str) -> Path:
        """
        Resolve the XML dump path for the requested session.
        """

        directory = self.__root / "xmls" / package_name / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def get_artifact_path(
        self,
        *,
        kind: ArtifactKind,
        package_name: str,
        session_id: str,
        filename: str,
    ) -> Path:
        """
        Resolve an artifact path under the temp root for the given kind.
        """

        directory = self.__root / kind.value / package_name / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename


class TestPerceptionServicePipelineBranch(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the pipeline-backed perceive path; the stable artifact identifier flows into both
    ``metadata["storage_id"]`` and ``capture.screenshot_uri`` without leaking an EFS path.
    """

    async def asyncSetUp(self) -> None:
        self.__tmp = TemporaryDirectory()
        self.__path_manager = _TempPathManager(root=Path(self.__tmp.name))

    async def asyncTearDown(self) -> None:
        self.__tmp.cleanup()

    def __build_pipeline(self, *, sink: ArtifactSinkPort) -> ArtifactPipeline:
        """
        Compose a pipeline whose only renderer covers ``SCREENSHOT``.
        """

        renderers: Mapping[ArtifactKind, PassthroughRenderer] = {
            ArtifactKind.SCREENSHOT: PassthroughRenderer(kind=ArtifactKind.SCREENSHOT),
        }
        return ArtifactPipeline(
            config=PipelineConfiguration(),
            renderers=renderers,
            sink=sink,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            workflow_id="workflow-1",
        )

    async def test_perceive_emits_single_screenshot_to_sink(self) -> None:
        """
        The pipeline receives exactly one SCREENSHOT record per perceive call, and the perception port is
        polled exactly once for the underlying capture.
        """

        sink = _RecordingSink()
        pipeline = self.__build_pipeline(sink=sink)
        perception = FakePerception(capture=_build_capture())
        service = PerceptionService(
            storage=FakeStorage(),
            perception=perception,
            hierarchy_signature_builder=Mock(),
            pipeline=pipeline,
        )

        await service.perceive(session_id="session-1", step_number=1)
        await pipeline.drain()

        self.assertEqual(perception.call_count, 1)
        self.assertEqual(
            [metadata.kind for metadata in sink.persisted],
            [ArtifactKind.SCREENSHOT],
        )

    async def test_perceive_stamps_screenshot_uri_with_efs_staged_path(self) -> None:
        """
        ``screenshot_uri`` carries the EFS-staged path returned by pipeline.emit so callers read bytes locally
        instead of hitting the sink-backed URL. The path must live under the path manager's temp root.
        """

        sink = _RecordingSink()
        pipeline = self.__build_pipeline(sink=sink)
        perception = FakePerception(capture=_build_capture())
        service = PerceptionService(
            storage=FakeStorage(),
            perception=perception,
            hierarchy_signature_builder=Mock(),
            pipeline=pipeline,
        )

        result = await service.perceive(session_id="session-1", step_number=3)
        await pipeline.drain()

        self.assertIsNotNone(result.screenshot_uri)
        assert result.screenshot_uri is not None
        self.assertTrue(result.screenshot_uri.startswith(self.__tmp.name))
        self.assertTrue(result.screenshot_uri.endswith(".png"))
        self.assertNotIn("cloud://", result.screenshot_uri)
