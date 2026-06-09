from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from fathom.adapters.artifact.cloud import CloudSink
from fathom.interfaces.storage import StoragePort
from fathom.schemas.artifact import ArtifactKind, ArtifactMetadata


class _RecordingStorage(StoragePort):
    """
    :class:`StoragePort` test double that records calls and returns a fake URL.
    """

    def __init__(self, *, identifier: str = "cloud://artifact/1") -> None:
        """
        Initialise the double with the identifier ``save`` should return.
        """

        self.__identifier = identifier
        self.calls: List[Dict[str, Any]] = []

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record the call and return the fake cloud identifier.
        """

        self.calls.append({"size": len(data), "metadata": dict(metadata or {})})
        return self.__identifier


class _RaisingStorage(StoragePort):
    """
    :class:`StoragePort` test double that always raises on upload.
    """

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Raise to exercise the failure path of the cloud sink.
        """

        _ = (data, metadata)
        raise RuntimeError("cloud unavailable")


class CloudSinkTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :class:`CloudSink` upload + receipt semantics.

    Success path returns ``local_cleanup=True`` so the pipeline removes
    the EFS staging files. Failure path returns ``local_cleanup=False`` so the next ``replay()`` can retry uploading.
    """

    @staticmethod
    def __metadata() -> ArtifactMetadata:
        """
        Minimal :class:`ArtifactMetadata` fixture identifying the artifact.
        """

        return ArtifactMetadata(
            kind=ArtifactKind.SCREENSHOT,
            created=1,
            step_number=0,
            package_name="app",
            session_id="run-test",
            filename="step-000__screenshot__2026-01-01T00-00-00Z-000.png",
        )

    async def test_successful_upload_returns_cleanup_receipt(self) -> None:
        """
        On a successful upload the cloud sink returns the cloud identifier and clears the local copy.
        """

        storage = _RecordingStorage(identifier="cloud://artifact/42")
        sink = CloudSink(storage=storage, workflow_id="run-test")

        receipt = await sink.persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertTrue(receipt.local_cleanup)
        self.assertEqual(len(storage.calls), 1)
        self.assertEqual(receipt.identifier, "cloud://artifact/42")
        self.assertEqual(storage.calls[0]["metadata"]["category"], "screenshot")

    async def test_failed_upload_keeps_local_copy_for_replay(self) -> None:
        """
        Any provider exception must surface as ``local_cleanup=False`` so the pipeline preserves the EFS file for the next replay scan.
        """

        sink = CloudSink(storage=_RaisingStorage(), workflow_id="run-test")

        receipt = await sink.persist(
            content=b"PNG",
            metadata=self.__metadata(),
        )

        self.assertFalse(receipt.local_cleanup)
        self.assertEqual(receipt.identifier, "")

    async def test_empty_identifier_is_treated_as_silent_failure(self) -> None:
        """
        When the underlying storage returns an empty identifier (the composite-storage signal for "all backends failed"),
        the sink treats it as a failure and preserves the EFS copy. This guards against the historical bug where a cloud backend was configured but artifacts silently went missing.
        """

        storage = _RecordingStorage(identifier="")
        sink = CloudSink(storage=storage, workflow_id="run-test")

        receipt = await sink.persist(
            metadata=self.__metadata(),
            content=b"PNG",
        )

        self.assertFalse(receipt.local_cleanup)
        self.assertEqual(receipt.identifier, "")

    async def test_canonical_filename_is_forwarded_to_storage(self) -> None:
        """
        The canonical ``step-NNN__kind__iso`` filename must flow into storage metadata.
        """

        storage = _RecordingStorage()
        sink = CloudSink(storage=storage, workflow_id="run-test")

        canonical = "step-007__screenshot__2026-06-08T01-02-03Z-456.png"
        metadata = ArtifactMetadata(
            kind=ArtifactKind.SCREENSHOT,
            session_id="run-test",
            package_name="app",
            step_number=7,
            created=1,
            filename=canonical,
        )
        await sink.persist(content=b"PNG", metadata=metadata)

        self.assertEqual(storage.calls[0]["metadata"]["filename"], canonical)


class CloudSinkPerKindCategoryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the per-:class:`ArtifactKind` category routing exposed via storage metadata.
    """

    @staticmethod
    def __metadata(*, kind: ArtifactKind) -> ArtifactMetadata:
        """
        Build a minimal :class:`ArtifactMetadata` parameterized on artifact kind.
        """

        return ArtifactMetadata(
            kind=kind,
            created=1,
            step_number=0,
            package_name="app",
            session_id="run-test",
            filename="step-000__screenshot__2026-01-01T00-00-00Z-000.png",
        )

    async def test_perception_kinds_route_to_annotated_category(self) -> None:
        """
        OCR/vision/overlay/cv/icon perception artifacts all land in the annotated GCS category.
        """

        cases = {
            ArtifactKind.CV_PERCEPTION: "annotated",
            ArtifactKind.OCR_PERCEPTION: "annotated",
            ArtifactKind.ICON_PERCEPTION: "annotated",
            ArtifactKind.VISION_PERCEPTION: "annotated",
            ArtifactKind.OVERLAY_PERCEPTION: "annotated",
        }
        for kind, expected_category in cases.items():
            with self.subTest(kind=kind):
                storage = _RecordingStorage()
                sink = CloudSink(storage=storage, workflow_id="run-test")

                await sink.persist(
                    content=b"PNG",
                    metadata=self.__metadata(kind=kind),
                )

                self.assertEqual(storage.calls[0]["metadata"]["category"], expected_category)
