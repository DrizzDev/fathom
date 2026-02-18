from __future__ import annotations

from fathom.adapters.device.adb import ADBDevice
from fathom.adapters.device.remote import RemoteDeviceAdapter
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import ADBConfiguration, DeviceConfiguration


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
