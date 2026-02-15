"""Fluent builder API for Fathom configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from fathom.interfaces.device import DevicePort
    from fathom.interfaces.knowledge import KnowledgePort
    from fathom.interfaces.llm import LLMPort
    from fathom.interfaces.memory import MemoryPort
    from fathom.interfaces.signal import SignalPort
    from fathom.interfaces.storage import StoragePort
    from fathom.interfaces.telemetry import TelemetryPort
    from fathom.runtime.runner import FathomRunner


class FathomBuilder:
    """
    Fluent builder for Fathom configuration.
    
    Methods are order-independent. Validation happens at build().
    """

    def __init__(self) -> None:
        """Initialize builder with no ports configured."""
        self._device: Optional[DevicePort] = None
        self._llm: Optional[LLMPort] = None
        self._memory: Optional[MemoryPort] = None
        self._knowledge: Optional[KnowledgePort] = None
        self._signal: Optional[SignalPort] = None
        self._storage: Optional[StoragePort] = None
        self._telemetry: Optional[TelemetryPort] = None

    def device(self, device: DevicePort) -> FathomBuilder:
        """Configure device port."""
        self._device = device
        return self

    def llm(self, llm: LLMPort) -> FathomBuilder:
        """Configure LLM port."""
        self._llm = llm
        return self

    def memory(self, memory: MemoryPort) -> FathomBuilder:
        """Configure memory port."""
        self._memory = memory
        return self

    def knowledge(self, knowledge: KnowledgePort) -> FathomBuilder:
        """Configure knowledge port."""
        self._knowledge = knowledge
        return self

    def signal(self, signal: SignalPort) -> FathomBuilder:
        """Configure signal port."""
        self._signal = signal
        return self

    def storage(self, storage: StoragePort) -> FathomBuilder:
        """Configure storage port."""
        self._storage = storage
        return self

    def telemetry(self, telemetry: TelemetryPort) -> FathomBuilder:
        """Configure telemetry port."""
        self._telemetry = telemetry
        return self

    def build(self) -> FathomRunner:
        """
        Build configured Fathom instance.
        
        Validates required ports and applies defaults.
        
        Returns:
            FathomRunner instance with all ports configured
            
        Raises:
            ValueError: If required ports (device, llm) are not configured
        """
        # Import here to avoid circular dependency
        from fathom.adapters.knowledge.sqlite import SQLiteKnowledge
        from fathom.adapters.memory.sqlite import SQLiteMemory
        from fathom.adapters.signal.noop import NoopSignal
        from fathom.adapters.storage.local import LocalStorage
        from fathom.adapters.telemetry.structlog import StructlogAdapter
        from fathom.runtime.runner import FathomRunner

        # Validate required ports
        if not self._device:
            raise ConfigurationError("Device port is required. Call .device() before .build()")
        if not self._llm:
            raise ConfigurationError("LLM port is required. Call .llm() before .build()")

        # Apply defaults for optional ports
        if not self._memory:
            self._memory = SQLiteMemory()
        if not self._knowledge:
            self._knowledge = SQLiteKnowledge()
        if not self._signal:
            self._signal = NoopSignal()
        if not self._storage:
            self._storage = LocalStorage()
        if not self._telemetry:
            self._telemetry = StructlogAdapter()

        return FathomRunner(
            device=self._device,
            llm=self._llm,
            memory=self._memory,
            knowledge=self._knowledge,
            signal=self._signal,
            storage=self._storage,
            telemetry=self._telemetry,
        )


class Fathom:
    """Main entry point for Fathom library."""

    @staticmethod
    def builder() -> FathomBuilder:
        """Create a new builder instance."""
        return FathomBuilder()
