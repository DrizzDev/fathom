from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from fathom.core.services.perception import PerceptionService
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.storage import StoragePort
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
        """Returned capture exposes the storage identifier via `metadata['storage_id']`."""

        storage = FakeStorage(storage_id="storage://artifact-id")
        perception = FakePerception(capture=_build_capture())
        service = PerceptionService(
            storage=storage,
            perception=perception,
            hierarchy_signature_builder=Mock(),
        )

        result = await service.perceive(session_id="session-1", step_number=1)

        self.assertEqual(result.metadata["storage_id"], "storage://artifact-id")
        # Non-local URIs must not be exposed as a filesystem `path`.
        self.assertNotIn("path", result.metadata)
