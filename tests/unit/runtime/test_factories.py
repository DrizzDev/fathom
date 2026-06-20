from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.adapters.device.remote.ios import IOSRemoteDeviceAdapter
from fathom.adapters.interaction.noop import NoopInteraction
from fathom.adapters.interaction.pypika.postgres import PostgresInteraction
from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.constants.storage import InteractionBackend
from fathom.core.exceptions import StorageConfigurationError
from fathom.runtime.factories import DeviceFactory, InteractionFactory
from fathom.schemas.configuration import (
    DeviceConfiguration,
    InteractionStorageConfiguration,
    NoopInteractionConfiguration,
    PostgresInteractionConfiguration,
    RemoteDeviceConfiguration,
    SQLiteInteractionConfiguration,
)


class DeviceFactoryTest(unittest.TestCase):
    """
    Keep Android and iOS remote device selection distinct.
    """

    def test_remote_ios_uses_ios_remote_adapter(self) -> None:
        """
        Select the iOS-specific remote adapter for remote iOS runs.
        """

        factory = DeviceFactory()
        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient"):
            device = factory.create(
                configuration=DeviceConfiguration(
                    type=DeviceConnectionType.REMOTE,
                    platform=DevicePlatform.IOS,
                    remote=RemoteDeviceConfiguration(
                        session_id="session-id",
                        provider_url="https://example.test",
                    ),
                )
            )

        self.assertIsInstance(device, IOSRemoteDeviceAdapter)

    def test_remote_android_uses_android_remote_adapter(self) -> None:
        """
        Keep Android remote runs on the transport-only adapter.
        """

        factory = DeviceFactory()
        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient"):
            device = factory.create(
                configuration=DeviceConfiguration(
                    type=DeviceConnectionType.REMOTE,
                    platform=DevicePlatform.ANDROID,
                    remote=RemoteDeviceConfiguration(
                        provider_url="https://example.test",
                        session_id="session-id",
                    ),
                )
            )

        self.assertIsInstance(device, ADBRemoteDeviceAdapter)


class InteractionFactoryTest(unittest.TestCase):
    """
    Verify backend dispatch and missing-configuration handling.
    """

    def setUp(self) -> None:
        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)
        self.__path = Path(self.__temporary_directory.name) / "interaction.db"

    def test_sqlite_backend_returns_sqlite_adapter(self) -> None:
        """
        Selecting the SQLite backend returns the SQLite adapter.
        """

        configuration = InteractionStorageConfiguration(
            backend=InteractionBackend.SQLITE,
            sqlite=SQLiteInteractionConfiguration(path=self.__path),
        )
        adapter = InteractionFactory().create(configuration=configuration)
        self.assertIsInstance(adapter, SQLiteInteraction)

    def test_noop_backend_returns_noop_adapter(self) -> None:
        """
        Selecting the noop backend returns the noop adapter.
        """

        configuration = InteractionStorageConfiguration(
            backend=InteractionBackend.NOOP,
            noop=NoopInteractionConfiguration(),
        )
        adapter = InteractionFactory().create(configuration=configuration)
        self.assertIsInstance(adapter, NoopInteraction)

    def test_postgres_backend_returns_postgres_adapter(self) -> None:
        """
        Selecting the Postgres backend returns the Postgres adapter.
        """

        configuration = InteractionStorageConfiguration(
            backend=InteractionBackend.POSTGRES,
            postgres=PostgresInteractionConfiguration(
                host="localhost",
                user="fathom",
                password="secret",
                database="fathom",
            ),
        )
        adapter = InteractionFactory().create(configuration=configuration)
        self.assertIsInstance(adapter, PostgresInteraction)

    def test_envelope_rejects_sqlite_backend_without_sqlite_config(self) -> None:
        """
        Envelope construction enforces matching nested configuration.
        """

        with self.assertRaises(ValueError):
            InteractionStorageConfiguration(backend=InteractionBackend.SQLITE)

    def test_envelope_rejects_postgres_backend_without_postgres_config(self) -> None:
        """
        Envelope construction enforces matching nested configuration.
        """

        with self.assertRaises(ValueError):
            InteractionStorageConfiguration(backend=InteractionBackend.POSTGRES)

    def test_envelope_rejects_noop_backend_without_noop_config(self) -> None:
        """
        Envelope construction enforces matching nested configuration.
        """

        with self.assertRaises(ValueError):
            InteractionStorageConfiguration(backend=InteractionBackend.NOOP)

    def test_postgres_pool_max_must_be_at_least_min(self) -> None:
        """
        Postgres pool sizing must be self-consistent at validation time.
        """

        with self.assertRaises(ValueError):
            PostgresInteractionConfiguration(
                host="x",
                user="fathom",
                password="secret",
                database="fathom",
                pool_min_size=10,
                pool_max_size=5,
            )

    def test_factory_emits_storage_error_on_explicit_missing_sqlite(self) -> None:
        """
        Bypass envelope validation by sending None and observe the typed error.
        """

        configuration = InteractionStorageConfiguration.model_construct(
            backend=InteractionBackend.SQLITE,
            sqlite=None,
            postgres=None,
            noop=None,
        )
        with self.assertRaises(StorageConfigurationError) as context:
            InteractionFactory().create(configuration=configuration)
        self.assertEqual(InteractionBackend.SQLITE.value, context.exception.backend)
