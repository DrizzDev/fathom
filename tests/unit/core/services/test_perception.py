from __future__ import annotations

import unittest
from pathlib import Path
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
    PipelineConfig,
)
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class FakeStorage(StoragePort):
    """In-memory storage that records every save call."""

    def __init__(self, *, storage_id: str = "/tmp/fathom/test/screenshot.png") -> None:
        self.__storage_id = storage_id
        self.calls: List[Dict[str, Any]] = []

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        self.calls.append({"data_len": len(data), "metadata": dict(metadata or {})})
        return self.__storage_id


class FakePerception(PerceptionPort):
    """Returns a pre-built capture; records call count."""

    def __init__(self, *, capture: ScreenCapture) -> None:
        self.__capture = capture
        self.call_count = 0

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        return None

    async def capture(self) -> ScreenCapture:
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


class TestPerceptionServicePersistCapture(unittest.IsolatedAsyncioTestCase):
    """Behavioural tests for `PerceptionService.perceive` storage metadata."""

    async def test_perceive_persists_screenshot_with_pre_action_phase(self) -> None:
        """Pre-action saves must carry `phase=pre_action` for downstream parity."""

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

    async def test_perceive_propagates_storage_id_into_capture_metadata(self) -> None:
        """Returned capture exposes the remote storage identifier via `metadata['storage_id']`."""

        storage = FakeStorage(storage_id="storage://artifact-id")
        perception = FakePerception(capture=_build_capture())
        service = PerceptionService(
            storage=storage,
            perception=perception,
            hierarchy_signature_builder=Mock(),
        )

        result = await service.perceive(session_id="session-1", step_number=1)

        self.assertEqual(result.metadata["storage_id"], "storage://artifact-id")
        # No filesystem path must ever leak through ScreenCapture metadata.
        self.assertNotIn("path", result.metadata)


class _RecordingSink(ArtifactSinkPort):
    """Sink that captures every persisted record without touching disk."""

    def __init__(self) -> None:
        self.persisted: List[ArtifactMetadata] = []

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
    ) -> ArtifactReceipt:
        self.persisted.append(metadata)
        return ArtifactReceipt(identifier="cloud://artifact", local_cleanup=True)


class _TempPathManager:
    """Minimal path manager surface stubbed against a temp directory root."""

    def __init__(self, *, root: Path) -> None:
        self.__root = root

    def get_xml_path(self, package_name: str, session_id: str, filename: str) -> Path:
        """Resolve a temp XML path for the requested session."""

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
        """Resolve a temp artifact path under the root."""

        directory = self.__root / kind.value / package_name / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename


class TestPerceptionServicePipelineBranch(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the pipeline-backed perceive path. Verifies the producer
    never leaks an Infrastructure-internal EFS staging path through
    Domain-visible ``ScreenCapture.metadata``.
    """

    async def asyncSetUp(self) -> None:
        from tempfile import TemporaryDirectory

        self.__tmp = TemporaryDirectory()
        self.__path_manager = _TempPathManager(root=Path(self.__tmp.name))

    async def asyncTearDown(self) -> None:
        self.__tmp.cleanup()

    def __build_pipeline(
        self,
        *,
        sink: ArtifactSinkPort,
    ) -> ArtifactPipeline:
        renderers: Mapping[ArtifactKind, PassthroughRenderer] = {
            ArtifactKind.SCREENSHOT: PassthroughRenderer(kind=ArtifactKind.SCREENSHOT),
        }
        return ArtifactPipeline(
            config=PipelineConfig(),
            renderers=renderers,
            sink=sink,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            workflow_id="workflow-1",
        )

    async def test_perceive_does_not_publish_filesystem_path_when_pipeline_is_wired(
        self,
    ) -> None:
        """
        The pipeline's EFS staging path must never appear in the
        returned capture's metadata; bytes-flow consumers rely only on
        ``capture.image``.
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

        result = await service.perceive(session_id="session-1", step_number=1)
        await pipeline.drain()

        self.assertNotIn("path", result.metadata)
        self.assertNotIn("storage_id", result.metadata)
        self.assertEqual(perception.call_count, 1)
        emitted_kinds = [metadata.kind for metadata in sink.persisted]
        self.assertEqual(emitted_kinds, [ArtifactKind.SCREENSHOT])
