from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import (
    DeviceConfiguration,
    LLMConfiguration,
    TelemetryConfiguration,
)


class DeviceFactoryPort(ABC):
    """
    Abstract factory contract for device adapter creation.
    """

    @abstractmethod
    def create(self, *, configuration: DeviceConfiguration) -> DevicePort:
        """
        Create device adapter from runtime device configuration.
        """

        raise NotImplementedError


class PerceptionFactoryPort(ABC):
    """
    Abstract factory contract for perception adapter creation.
    """

    @abstractmethod
    def create(
        self,
        *,
        configuration: DeviceConfiguration,
        device: DevicePort,
        use_xml: bool,
        capture_hierarchy: bool = False,
    ) -> PerceptionPort:
        """
        Create perception adapter from runtime configuration and bound ports.
        """

        raise NotImplementedError


class LLMFactoryPort(ABC):
    """
    Abstract factory contract for LLM adapter creation.
    """

    @abstractmethod
    def create(self, *, configuration: LLMConfiguration) -> LLMPort:
        """
        Create LLM adapter from runtime LLM configuration.
        """

        raise NotImplementedError


class SignalFactoryPort(ABC):
    """
    Abstract factory contract for signal adapter creation.
    """

    @abstractmethod
    def create(
        self,
        *,
        signal_type: str,
        interactive: bool,
        workflow_id: Optional[str] = None,
    ) -> SignalPort:
        """
        Create signal adapter from runtime interaction mode.
        """

        raise NotImplementedError


class TelemetryFactoryPort(ABC):
    """
    Abstract factory contract for telemetry adapter creation.
    """

    @abstractmethod
    def create(self, *, configuration: TelemetryConfiguration) -> TelemetryPort:
        """
        Create telemetry adapter from runtime telemetry configuration.
        """

        raise NotImplementedError
