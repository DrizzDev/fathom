from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple
from unittest import mock

from PIL import Image

from fathom.adapters.artifact.cloud import CloudSink
from fathom.adapters.artifact.noop import NoopSink
from fathom.base.paths import SharedPathManager
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.artifact.renderer import (
    PassthroughRenderer,
    PerceptionRenderer,
)
from fathom.interfaces.artifact import ArtifactRendererPort, ArtifactSinkPort
from fathom.interfaces.storage import StoragePort
from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactMetadata,
    ArtifactReceipt,
    ArtifactRecord,
    LocalArtifactPolicy,
    OcrRawPayload,
    PipelineConfiguration,
    ScreenshotPayload,
)
from fathom.schemas.screens import ScreenCapture
from fathom.settings.env import FathomSettings


class _BadRenderer(ArtifactRendererPort):
    """
    Renderer that always raises so the pipeline's failure path can be pinned.
    """

    @property
    def kind(self) -> ArtifactKind:
        """
        Bind this renderer to the screenshot kind for routing.
        """

        return ArtifactKind.SCREENSHOT

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Raise unconditionally to drive the render-failure branch.
        """

        _ = record
        raise RuntimeError("renderer down")


class _RecordingSink(ArtifactSinkPort):
    """
    :class:`ArtifactSinkPort` test double tracking every persist call.
    """

    def __init__(self, *, cleanup: bool, raise_error: bool = False) -> None:
        """
        Initialise the double with the cleanup hint to report and the
        optional error-injection flag for the failure-path test.
        """

        self.calls: List[Tuple[ArtifactMetadata, bytes]] = []
        self.__cleanup = cleanup
        self.__raise = raise_error

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
    ) -> ArtifactReceipt:
        """
        Record the call and return the configured receipt.
        """

        self.calls.append((metadata, content))
        if self.__raise:
            raise RuntimeError("sink failure")
        return ArtifactReceipt(
            identifier="recorded" if self.__cleanup else "kept",
            local_cleanup=self.__cleanup,
        )


class _NullStorage(StoragePort):
    """
    Trivial :class:`StoragePort` returning a synthetic identifier.
    """

    async def save(self, *, data, metadata=None):  # type: ignore[no-untyped-def]
        """
        Pretend the upload succeeded and surface a stable identifier.
        """

        _ = (data, metadata)
        return "cloud://artifact/test"


def _path_manager(*, tmp: Path) -> SharedPathManager:
    """
    Build a :class:`SharedPathManager` rooted at a temporary directory.

    Wrapped as a closure so each test gets isolation; the underlying
    settings object only needs ``assets_path``.
    """

    settings = mock.create_autospec(FathomSettings, instance=True)
    settings.assets_path = tmp
    return SharedPathManager(settings=settings)


def _png_bytes() -> bytes:
    """
    Encode a deterministic 4x4 PNG used as the capture image fixture.
    """

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _record(*, step: int = 0, created: int = 1) -> ArtifactRecord:
    """
    Build an :class:`ArtifactRecord` carrying a screenshot payload.
    """

    return ArtifactRecord(
        session_id="run-test",
        package_name="app",
        step_number=step,
        created=created,
        payload=ScreenshotPayload(
            capture=ScreenCapture(
                width=4,
                height=4,
                activity="app",
                image=_png_bytes(),
                timestamp=0,
            ),
        ),
    )


def _pipeline(
    *,
    tmp: Path,
    sink: ArtifactSinkPort,
    config: PipelineConfiguration | None = None,
) -> ArtifactPipeline:
    """
    Wire a :class:`ArtifactPipeline` for tests with a passthrough renderer.
    """

    return ArtifactPipeline(
        config=config or PipelineConfiguration(),
        renderers={
            ArtifactKind.SCREENSHOT: PassthroughRenderer(kind=ArtifactKind.SCREENSHOT),
            ArtifactKind.PERCEPTION: PerceptionRenderer(),
        },
        sink=sink,
        path_manager=_path_manager(tmp=tmp),
        workflow_id="run-test",
    )


class ArtifactPipelineStagingTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the synchronous EFS-staging behaviour of :meth:`ArtifactPipeline.emit`.

    The staging write IS the durability boundary — emit must not return
    until the payload bytes are on disk.
    """

    async def test_emit_writes_payload_to_efs(self) -> None:
        """
        After ``emit`` returns the payload exists at the resolved path.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=False)
            pipeline = _pipeline(tmp=tmp_path, sink=sink)

            await pipeline.emit(record=_record())
            await pipeline.drain()

            screenshot_dir = tmp_path / "screenshot"
            payloads = list(screenshot_dir.rglob("step-000__screenshot__*.png"))
            self.assertEqual(len(payloads), 1)
            self.assertTrue(payloads[0].read_bytes())

    async def test_emit_writes_ocr_raw_json_to_xmls_directory(self) -> None:
        """
        Raw OCR artifacts share the hierarchy/XML directory and use .json.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=False)
            pipeline = ArtifactPipeline(
                config=PipelineConfiguration(),
                renderers={
                    ArtifactKind.OCR_RAW: PassthroughRenderer(kind=ArtifactKind.OCR_RAW),
                },
                sink=sink,
                path_manager=_path_manager(tmp=tmp_path),
                workflow_id="run-test",
            )

            await pipeline.emit(
                record=ArtifactRecord(
                    session_id="run-test",
                    package_name="app",
                    step_number=0,
                    created=1_700_000_000_000,
                    payload=OcrRawPayload(content='{"text": "Swiggy"}'),
                )
            )
            await pipeline.drain()

            payloads = list(tmp_path.rglob("xmls/**/step-000__ocr_raw__*.json"))
            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0].read_text(), '{"text": "Swiggy"}')

    async def test_emit_drops_record_when_kind_has_no_renderer(self) -> None:
        """
        Records whose kind is not in the renderers map are dropped and
        no EFS file is written.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=False)
            pipeline = ArtifactPipeline(
                config=PipelineConfiguration(),
                renderers={},
                sink=sink,
                path_manager=_path_manager(tmp=tmp_path),
                workflow_id="run-test",
            )

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertEqual(list(tmp_path.rglob("*")), [])
            self.assertEqual(sink.calls, [])

    async def test_emit_swallows_renderer_failures(self) -> None:
        """
        A renderer that raises must not propagate; the record is dropped
        and no EFS file is written.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=False)
            pipeline = ArtifactPipeline(
                config=PipelineConfiguration(),
                renderers={ArtifactKind.SCREENSHOT: _BadRenderer()},
                sink=sink,
                path_manager=_path_manager(tmp=tmp_path),
                workflow_id="run-test",
            )

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertEqual(sink.calls, [])


class ArtifactPipelineCleanupTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins post-upload cleanup driven by the sink's :class:`ArtifactReceipt`.
    """

    async def test_cleanup_true_unlinks_payload(self) -> None:
        """
        ``local_cleanup=True`` causes the EFS file to be removed.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=True)
            pipeline = _pipeline(tmp=tmp_path, sink=sink)

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertEqual(list(tmp_path.rglob("*.png")), [])

    async def test_cleanup_false_keeps_efs_files(self) -> None:
        """
        ``local_cleanup=False`` leaves the EFS file in place.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=False)
            pipeline = _pipeline(tmp=tmp_path, sink=sink)

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertTrue(list(tmp_path.rglob("*.png")))

    async def test_sink_exception_keeps_efs_files(self) -> None:
        """
        A sink that raises must not crash the pipeline; the EFS file
        survives so the local copy remains usable.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=True, raise_error=True)
            pipeline = _pipeline(tmp=tmp_path, sink=sink)

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertTrue(list(tmp_path.rglob("*.png")))


class ArtifactPipelineLocalRetentionPolicyTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the ``PipelineConfiguration.local.cleanup`` gate.

    Hosts that consume the EFS-staged path after the sink ack disable
    this flag and own a fallback sweep; the pipeline must skip the
    unlink whenever the policy is off, even if the sink reports
    cleanup-safe.
    """

    async def test_policy_off_keeps_efs_when_sink_acks_cleanup(self) -> None:
        """
        Policy disabled overrides a sink receipt's ``local_cleanup=True``.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=True)
            pipeline = _pipeline(
                tmp=tmp_path,
                sink=sink,
                config=PipelineConfiguration(local=LocalArtifactPolicy(cleanup=False)),
            )

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertTrue(list(tmp_path.rglob("*.png")))

    async def test_policy_on_preserves_default_cleanup(self) -> None:
        """
        Default policy ``cleanup=True`` still unlinks on a clean sink ack.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=True)
            pipeline = _pipeline(
                tmp=tmp_path,
                sink=sink,
                config=PipelineConfiguration(local=LocalArtifactPolicy(cleanup=True)),
            )

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertEqual(list(tmp_path.rglob("*.png")), [])


class ArtifactPipelineCloudIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """
    Smoke pin that the pipeline routes records through a real
    :class:`CloudSink` and clears local state on success.
    """

    async def test_cloud_round_trip_clears_local(self) -> None:
        """
        End-to-end: emit → render → stage → cloud upload → local cleanup.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cloud = CloudSink(storage=_NullStorage(), workflow_id="run-test")
            pipeline = _pipeline(tmp=tmp_path, sink=cloud)

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertEqual(list(tmp_path.rglob("*.png")), [])


class ArtifactPipelineDrainTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins drain semantics so survivors are left on EFS rather than lost in memory.
    """

    async def test_drain_with_no_pending_returns_immediately(self) -> None:
        """
        Drain on an idle pipeline is a no-op.
        """

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _pipeline(tmp=Path(tmp), sink=NoopSink())

            await pipeline.drain()
            self.assertEqual(pipeline.pending_count, 0)

    async def test_drain_completes_after_pending_finish(self) -> None:
        """
        After ``drain`` returns there are no pending tasks left.
        """

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sink = _RecordingSink(cleanup=False)
            pipeline = _pipeline(tmp=tmp_path, sink=sink)

            await pipeline.emit(record=_record())
            await pipeline.drain()

            self.assertEqual(pipeline.pending_count, 0)
            self.assertEqual(len(sink.calls), 1)
