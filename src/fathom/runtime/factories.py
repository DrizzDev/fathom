from __future__ import annotations

from fathom.adapters.device.adb import ADBDevice
from fathom.adapters.device.remote import RemoteDeviceAdapter
from fathom.adapters.storage.cloud import CloudStorage
from fathom.adapters.telemetry.redis import RedisTelemetryAdapter
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import (
    ADBConfiguration,
    DeviceConfiguration,
    StorageConfiguration,
    TelemetryConfiguration,
)


class DeviceFactory:
    """
    Factory for creating device adapters based on configuration.
    """

    @staticmethod
    def create(configuration: DeviceConfiguration) -> DevicePort:
        """
        Creates the appropriate DevicePort implementation.
        """

        if configuration.type == "REMOTE":
            return RemoteDeviceAdapter(configuration=configuration)
        else:
            return ADBDevice(
                configuration=ADBConfiguration(serial_number=configuration.serial_number)
            )


class TelemetryFactory:
    """
    Factory for creating telemetry adapters based on configuration.
    """

    @staticmethod
    def create(configuration: TelemetryConfiguration) -> TelemetryPort:
        """
        Creates the appropriate TelemetryPort implementation.
        """

        if configuration.type == "REDIS":
            return RedisTelemetryAdapter(configuration=configuration)
        else:
            return StructlogAdapter()


class StorageFactory:
    """
    Factory for creating storage adapters based on configuration.
    """

    @staticmethod
    def create(configuration: StorageConfiguration, path_manager: SharedPathManager) -> StoragePort:
        """
        Creates the appropriate StoragePort implementation.
        """

        from fathom.adapters.storage.composite import CompositeStorage
        from fathom.adapters.storage.local import LocalStorage

        storages: list[StoragePort] = []

        if "LOCAL" in configuration.backends:
            storages.append(LocalStorage(path_manager=path_manager))

        if "CLOUD" in configuration.backends and configuration.storage_bucket:
            storages.append(CloudStorage(storage=GCSImageStorage(configuration=configuration)))

        if storages:
            return CompositeStorage(storages=storages)

        raise ValueError("Please provide at-least one backend for storage")
