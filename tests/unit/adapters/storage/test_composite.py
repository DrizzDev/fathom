from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

from fathom.adapters.storage.composite import CompositeStorage
from fathom.interfaces.storage import StoragePort


class _SuccessfulStorage(StoragePort):
    """
    :class:`StoragePort` test double that returns a fixed identifier.
    """

    def __init__(self, *, identifier: str) -> None:
        """
        Bind the double to the identifier ``save`` should return.
        """

        self.__identifier = identifier
        self.calls: List[Dict[str, Any]] = []

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record the call and return the fixed identifier.
        """

        self.calls.append({"size": len(data), "metadata": dict(metadata or {})})
        return self.__identifier


class _RaisingStorage(StoragePort):
    """
    :class:`StoragePort` test double that always raises on upload.
    """

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Raise an exception to exercise the failure path.
        """

        _ = (data, metadata)
        raise RuntimeError("backend unavailable")


class CompositeStorageTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins composite-storage routing + the silent-loss observability surface.
    """

    async def test_returns_first_successful_identifier(self) -> None:
        """
        When at least one backend succeeds the composite returns its identifier.
        """

        primary = _SuccessfulStorage(identifier="primary://1")
        secondary = _SuccessfulStorage(identifier="secondary://1")
        composite = CompositeStorage(storages=[primary, secondary])

        identifier = await composite.save(data=b"x", metadata={"category": "screenshot"})

        self.assertEqual(identifier, "primary://1")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(secondary.calls), 1)

    async def test_empty_storage_list_returns_empty_with_warning(self) -> None:
        """
        A composite with no backends warns and returns ``""``.
        """

        with self.assertLogs("fathom.adapters.storage.composite", level="WARNING") as captured:
            identifier = await CompositeStorage(storages=[]).save(data=b"x")

        self.assertEqual(identifier, "")
        self.assertTrue(
            any("no backends configured" in record.getMessage() for record in captured.records),
            msg="expected a no-backends warning",
        )

    async def test_all_backends_failing_logs_error(self) -> None:
        """
        When every backend raises, the composite returns ``""`` and emits
        an error log explicitly identifying the silent-loss condition.
        """

        composite = CompositeStorage(
            storages=[_RaisingStorage(), _RaisingStorage()],
        )

        with self.assertLogs("fathom.adapters.storage.composite", level="ERROR") as captured:
            identifier = await composite.save(data=b"x", metadata={"category": "screenshot"})

        self.assertEqual(identifier, "")
        self.assertTrue(
            any("artifact lost" in record.getMessage() for record in captured.records),
            msg="expected the artifact-lost error log",
        )

    async def test_partial_failure_returns_surviving_identifier_with_warning(self) -> None:
        """
        Mixed success/failure surfaces a warning but still returns the
        successful identifier so the lifecycle proceeds.
        """

        composite = CompositeStorage(
            storages=[_RaisingStorage(), _SuccessfulStorage(identifier="cloud://ok")],
        )

        with self.assertLogs("fathom.adapters.storage.composite", level="WARNING") as captured:
            identifier = await composite.save(data=b"x")

        self.assertEqual(identifier, "cloud://ok")
        partial_logs = [
            record for record in captured.records if "partially succeeded" in record.getMessage()
        ]
        self.assertTrue(partial_logs, msg="expected a partial-success warning")
