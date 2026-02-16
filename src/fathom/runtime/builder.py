"""Fluent builder API for Fathom configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.core.exceptions import ConfigurationError
from fathom.schemas.configuration import (
    ExecutionConfig,
    ExplorationStrategyConfig,
    FathomConfig,
    IntentStrategyConfig,
)

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
        self._config: FathomConfig = FathomConfig()

    def device(self, device: DevicePort) -> FathomBuilder:
        """
        Configure device port (plug-and-play).
        
        Accepts any implementation of DevicePort interface:
        - ADBDevice: Android Debug Bridge
        - Custom: Your own device implementation
        
        Args:
            device: Device port implementation
        
        Returns:
            Self for method chaining
        """
        self._device = device
        return self

    def llm(self, llm: LLMPort) -> FathomBuilder:
        """
        Configure LLM port (plug-and-play).
        
        Accepts any implementation of LLMPort interface:
        - GeminiLLM: Google Gemini
        - OpenAILLM: OpenAI GPT
        - Custom: Your own LLM implementation
        
        Args:
            llm: LLM port implementation
        
        Returns:
            Self for method chaining
        """
        self._llm = llm
        return self

    def memory(self, memory: MemoryPort) -> FathomBuilder:
        """
        Configure memory port (plug-and-play).
        
        Accepts any implementation of MemoryPort interface:
        - SQLiteMemory: Local SQLite storage (default)
        - RedisMemory: Redis-based storage
        - Custom: Your own memory implementation
        
        Args:
            memory: Memory port implementation
        
        Returns:
            Self for method chaining
        """
        self._memory = memory
        return self

    def knowledge(self, knowledge: KnowledgePort) -> FathomBuilder:
        """
        Configure knowledge port (plug-and-play).
        
        Accepts any implementation of KnowledgePort interface:
        - SQLiteKnowledge: Local SQLite storage (default)
        - VectorKnowledge: Vector database storage
        - Custom: Your own knowledge implementation
        
        Args:
            knowledge: Knowledge port implementation
        
        Returns:
            Self for method chaining
        """
        self._knowledge = knowledge
        return self

    def signal(self, signal: SignalPort) -> FathomBuilder:
        """
        Configure signal port (plug-and-play).
        
        Accepts any implementation of SignalPort interface:
        - InteractiveSignal: Terminal-based HITL
        - TemporalSignalAdapter: Temporal workflow signals
        - NoopSignal: No HITL (default)
        - Custom: Your own signal implementation
        
        Args:
            signal: Signal port implementation
        
        Returns:
            Self for method chaining
        """
        self._signal = signal
        return self

    def storage(self, storage: StoragePort) -> FathomBuilder:
        """
        Configure storage port (plug-and-play).
        
        Accepts any implementation of StoragePort interface:
        - LocalStorage: Local filesystem storage (default)
        - S3Storage: AWS S3 storage
        - Custom: Your own storage implementation
        
        Args:
            storage: Storage port implementation
        
        Returns:
            Self for method chaining
        """
        self._storage = storage
        return self

    def telemetry(self, telemetry: TelemetryPort) -> FathomBuilder:
        """
        Configure telemetry port (plug-and-play).
        
        Accepts any implementation of TelemetryPort interface:
        - StructlogAdapter: Structured logging (default)
        - DatadogAdapter: Datadog APM
        - Custom: Your own telemetry implementation
        
        Args:
            telemetry: Telemetry port implementation
        
        Returns:
            Self for method chaining
        """
        self._telemetry = telemetry
        return self

    def config(self, config: FathomConfig) -> FathomBuilder:
        """Configure Fathom settings."""
        self._config = config
        return self

    def execution_config(self, config: ExecutionConfig) -> FathomBuilder:
        """Configure execution engine settings."""
        self._config.execution = config
        return self

    def intent_config(self, config: IntentStrategyConfig) -> FathomBuilder:
        """Configure intent strategy settings."""
        self._config.intent_strategy = config
        return self

    def exploration_config(self, config: ExplorationStrategyConfig) -> FathomBuilder:
        """Configure exploration strategy settings."""
        self._config.exploration_strategy = config
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
            config=self._config,
        )


class Fathom:
    """Main entry point for Fathom library."""

    @staticmethod
    def builder() -> FathomBuilder:
        """Create a new builder instance."""
        return FathomBuilder()
