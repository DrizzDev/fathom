from __future__ import annotations

from fathom.adapters.device.adb import ADBDevice
from fathom.adapters.device.remote import RemoteDeviceAdapter
from fathom.adapters.telemetry.redis import RedisTelemetryAdapter
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.interfaces.device import DevicePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import (
    ADBConfiguration,
    DeviceConfiguration,
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
