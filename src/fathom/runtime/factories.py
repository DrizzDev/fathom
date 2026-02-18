from __future__ import annotations

from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import DeviceConfiguration


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
            from fathom.adapters.device.remote import RemoteDeviceAdapter

            return RemoteDeviceAdapter(configuration=configuration)
        else:
            from fathom.adapters.device.adb import ADBDevice
            from fathom.schemas.configuration import ADBConfiguration

            # Map DeviceConfiguration to ADBConfiguration for local devices
            return ADBDevice(configuration=ADBConfiguration(serial_number=configuration.serial))
