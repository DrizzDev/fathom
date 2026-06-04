from __future__ import annotations

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch

from PIL import Image

from fathom.constants import ActionType
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.artifact.renderer import PassthroughRenderer
from fathom.core.services.hierarchy import HierarchyService
from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactMetadata,
    ArtifactReceipt,
    PipelineConfiguration,
)
from fathom.schemas.screens import ScreenCapture

_ANDROID_HIERARCHY = """
<hierarchy package="com.test.app">
  <node class="android.widget.Button" bounds="[100,200][500,400]" clickable="true" text="Primary" />
  <node class="android.widget.TextView" bounds="[100,500][800,600]" clickable="false" text="Title" />
  <node class="android.widget.EditText" bounds="[100,700][800,820]" clickable="true" text="" />
</hierarchy>
"""


def _png_bytes(width: int = 1080, height: int = 2400) -> bytes:
    """
    Render a solid-colour PNG for capture fixtures.
    """

    canvas = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _capture() -> ScreenCapture:
    """
    Build a frozen :class:`ScreenCapture` carrying valid PNG bytes.
    """

    return ScreenCapture(
        width=1080,
        height=2400,
        activity="com.test.app",
        image=_png_bytes(),
        timestamp=1_714_200_000_000,
        metadata={},
    )


class _UnlinkingSink(ArtifactSinkPort):
    """
    Sink that unlinks the EFS staging file the moment it sees the upload,
    exactly mirroring the prod CloudSink failure mode this fix addresses.

    Records every call so the test can assert the pipeline drove emit
    correctly for each artifact kind.
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
        Record the persist invocation and report ``local_cleanup=True``
        so the pipeline's unlink path runs synchronously in the test.
        """

        self.persisted.append(metadata)
        return ArtifactReceipt(identifier="local-sink", local_cleanup=True)


class _FakePathManager:
    """
    Minimal stand-in providing only the path-manager surface
    :class:`HierarchyService.process_xml_and_screen` exercises.
    """

    def __init__(self, *, root: Path) -> None:
        self.__root = root

    def get_xml_path(self, *, session_id: str, filename: str) -> Path:
        """
        Resolve the XML dump path for the requested session.
        """

        directory = self.__root / "xmls" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def get_artifact_path(
        self,
        *,
        kind: ArtifactKind,
        session_id: str,
        filename: str,
    ) -> Path:
        """
        Resolve an artifact path under the temp root.
        """

        directory = self.__root / kind.value / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename


class HierarchyServiceBytesContractTest(unittest.IsolatedAsyncioTestCase):
    """
    Behavioural pins for :class:`HierarchyService` after the bytes-flow refactor.
    """

    async def asyncSetUp(self) -> None:
        self.__tmp = TemporaryDirectory()
        self.__path_manager = _FakePathManager(root=Path(self.__tmp.name))

    async def asyncTearDown(self) -> None:
        self.__tmp.cleanup()

    def __build_pipeline(self, *, sink: ArtifactSinkPort) -> ArtifactPipeline:
        """
        Compose a pipeline with passthrough renderers for the two kinds
        :class:`HierarchyService` emits.
        """

        renderers = {
            ArtifactKind.HIERARCHY_XML: PassthroughRenderer(kind=ArtifactKind.HIERARCHY_XML),
            ArtifactKind.ANNOTATED: PassthroughRenderer(kind=ArtifactKind.ANNOTATED),
        }
        return ArtifactPipeline(
            config=PipelineConfiguration(),
            renderers=renderers,
            sink=sink,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            workflow_id="workflow-1",
        )

    async def test_extract_elements_consumes_screen_bytes_directly(self) -> None:
        """
        ``extract_elements`` must operate on ``screen.image`` without resolving any path.
        """

        service = HierarchyService()
        elements = service.extract_elements(
            xml=_ANDROID_HIERARCHY,
            screen=_capture(),
            action_type=ActionType.TAP,
        )

        self.assertGreater(len(elements), 0)

    async def test_process_xml_and_screen_returns_annotated_capture_with_bytes(self) -> None:
        """
        End-to-end happy path must produce labeled elements, a populated
        label map, and an annotated capture carrying non-empty PNG bytes.
        """

        sink = _UnlinkingSink()
        pipeline = self.__build_pipeline(sink=sink)
        service = HierarchyService(pipeline=pipeline)

        result = await service.process_xml_and_screen(
            _ANDROID_HIERARCHY,
            _capture(),
            session_id="session-1",
            package_name="com.test.app",
            step_number=3,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            action_type=ActionType.TAP,
        )
        await pipeline.drain()

        self.assertIsNotNone(result.annotated_capture)
        self.assertGreater(len(result.labeled_elements), 0)
        self.assertGreater(len(result.label_map), 0)
        self.assertIsNotNone(result.annotated_capture.annotated_image)
        self.assertGreater(len(result.annotated_capture.annotated_image or b""), 0)

    async def test_process_xml_and_screen_survives_pipeline_local_cleanup(self) -> None:
        """
        Race regression: even when the pipeline's sink completes upload
        and the staging file becomes eligible for cleanup before any
        downstream stage runs, the hierarchy result must still carry the
        labeled manifest. This pins the bytes-flow guarantee that
        Application is independent of the pipeline's EFS lifecycle.
        """

        sink = _UnlinkingSink()
        pipeline = self.__build_pipeline(sink=sink)
        service = HierarchyService(pipeline=pipeline)

        result = await service.process_xml_and_screen(
            _ANDROID_HIERARCHY,
            _capture(),
            session_id="session-1",
            package_name="com.test.app",
            step_number=7,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            action_type=ActionType.TAP,
        )
        await pipeline.drain()

        self.assertGreater(len(result.labeled_elements), 0)
        self.assertGreater(len(result.label_map), 0)
        emitted_kinds = {metadata.kind for metadata in sink.persisted}
        self.assertIn(ArtifactKind.HIERARCHY_XML, emitted_kinds)
        self.assertIn(ArtifactKind.ANNOTATED, emitted_kinds)

    async def test_process_xml_and_screen_returns_unannotated_fallback_when_annotator_fails(
        self,
    ) -> None:
        """
        Annotator failure must degrade to the original capture with the
        labeled manifest preserved — no exception leaks to the caller.
        """

        service = HierarchyService()
        original = _capture()

        with patch(
            "fathom.core.services.hierarchy.ImageAnnotator.annotate",
            return_value=None,
        ):
            result = await service.process_xml_and_screen(
                _ANDROID_HIERARCHY,
                original,
                session_id="session-1",
                package_name="com.test.app",
                step_number=2,
                path_manager=self.__path_manager,  # type: ignore[arg-type]
                action_type=ActionType.TAP,
            )

        self.assertGreater(len(result.labeled_elements), 0)
        self.assertIs(result.annotated_capture, original)

    async def test_process_xml_and_screen_without_pipeline_skips_emission(self) -> None:
        """
        Running without an artifact pipeline still produces the manifest
        and annotated capture; emission seams short-circuit gracefully.
        """

        service = HierarchyService()

        result = await service.process_xml_and_screen(
            _ANDROID_HIERARCHY,
            _capture(),
            session_id="session-1",
            package_name="com.test.app",
            step_number=4,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            action_type=ActionType.TAP,
        )

        self.assertIsNotNone(result.annotated_capture)
        self.assertGreater(len(result.labeled_elements), 0)
        self.assertIsNotNone(result.annotated_capture.annotated_image)

    async def test_process_xml_and_screen_returns_empty_result_when_xml_parse_raises(
        self,
    ) -> None:
        """
        A raised exception inside the processing block must be swallowed
        with logging and a clean empty result returned — never propagate.
        """

        service = HierarchyService()
        original = _capture()

        with patch(
            "fathom.core.services.hierarchy.BoundsGenerator.create_element",
            side_effect=RuntimeError("parser crashed"),
        ):
            result = await service.process_xml_and_screen(
                _ANDROID_HIERARCHY,
                original,
                session_id="session-1",
                package_name="com.test.app",
                step_number=5,
                path_manager=self.__path_manager,  # type: ignore[arg-type]
                action_type=ActionType.TAP,
            )

        self.assertEqual(result.labeled_elements, [])
        self.assertEqual(result.label_map, {})
        self.assertIs(result.annotated_capture, original)

    async def test_process_xml_and_screen_stamps_annotated_uri_from_pipeline_staged_path(
        self,
    ) -> None:
        """
        After the annotated bytes land in the pipeline, the staged path
        is stamped onto the annotated capture's typed ``annotated_uri``
        field as the canonical handle.
        """

        sink = _UnlinkingSink()
        pipeline = self.__build_pipeline(sink=sink)
        service = HierarchyService(pipeline=pipeline)

        result = await service.process_xml_and_screen(
            _ANDROID_HIERARCHY,
            _capture(),
            session_id="session-1",
            package_name="com.test.app",
            step_number=4,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            action_type=ActionType.TAP,
        )
        await pipeline.drain()

        self.assertIsNotNone(result.annotated_capture)
        self.assertIsNotNone(result.annotated_capture.annotated_uri)
        self.assertTrue(str(result.annotated_capture.annotated_uri).endswith(".png"))

    async def test_process_xml_and_screen_leaves_annotated_uri_unset_when_pipeline_missing(
        self,
    ) -> None:
        """
        Without an artifact pipeline the annotated bytes still flow
        through, but ``annotated_uri`` stays unset because no staged
        path exists to stamp.
        """

        service = HierarchyService()

        result = await service.process_xml_and_screen(
            _ANDROID_HIERARCHY,
            _capture(),
            session_id="session-1",
            package_name="com.test.app",
            step_number=4,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            action_type=ActionType.TAP,
        )

        self.assertIsNone(result.annotated_capture.annotated_uri)

    async def test_process_xml_and_screen_short_circuits_on_undersized_xml(self) -> None:
        """
        Tiny XML payloads must bypass parsing and return the original
        capture with empty manifest — no crash, no exception.
        """

        service = HierarchyService()
        original = _capture()

        result = await service.process_xml_and_screen(
            "<hierarchy/>",
            original,
            session_id="session-1",
            package_name="com.test.app",
            step_number=1,
            path_manager=self.__path_manager,  # type: ignore[arg-type]
            action_type=ActionType.TAP,
        )

        self.assertEqual(result.labeled_elements, [])
        self.assertEqual(result.label_map, {})
        self.assertIs(result.annotated_capture, original)
