from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, cast

from fathom.adapters.knowledge.sqlite import SQLiteKnowledge
from fathom.adapters.memory.sqlite import SQLiteMemory
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.storage.local import LocalStorage
from fathom.adapters.summarization.llm import LLMSummarizer
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.core.config.loader import RuntimeConfigLoader
from fathom.core.exceptions import ConfigurationError
from fathom.core.services.qualifier import IntentQualifierFactory
from fathom.infrastructure.memory.ledger import Ledger
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.runtime.factories import LLMFactory
from fathom.schemas.configuration import (
    ExecutionConfiguration,
    ExplorationConfiguration,
    FathomConfiguration,
    IntentConfiguration,
    LLMConfiguration,
    QualifierConfiguration,
    StorageConfiguration,
)
from fathom.schemas.run import RealignmentPolicy
from fathom.settings.env import FathomSettings

if TYPE_CHECKING:
    from fathom.interfaces.device import DevicePort
    from fathom.interfaces.factory import LLMFactoryPort
    from fathom.interfaces.interaction import InteractionPort
    from fathom.interfaces.knowledge import KnowledgePort
    from fathom.interfaces.llm import LLMPort
    from fathom.interfaces.memory import MemoryPort
    from fathom.interfaces.perception import PerceptionPort
    from fathom.interfaces.qualifier import IntentQualifierPort
    from fathom.interfaces.signal import SignalPort
    from fathom.interfaces.storage import StoragePort
    from fathom.interfaces.summarization import SummarizationPort
    from fathom.interfaces.telemetry import TelemetryPort
    from fathom.runtime.assembly import RunAssemblyBuilder
    from fathom.runtime.runner import FathomRunner


class FathomBuilder:
    """
    Fluent builder for Fathom configuration.

    Methods are order-independent. Validation happens at build().
    """

    def __init__(self, path_manager: Optional[SharedPathManager] = None) -> None:
        """Initialize the builder with no ports configured."""

        self.__device: Optional[DevicePort] = None
        self.__perception: Optional[PerceptionPort] = None

        self.__llm: Optional[LLMPort] = None
        self.__memory: Optional[MemoryPort] = None
        self.__interaction: Optional[InteractionPort] = None
        self.__knowledge: Optional[KnowledgePort] = None

        self.__signal: Optional[SignalPort] = None
        self.__storage: Optional[StoragePort] = None
        self.__telemetry: Optional[TelemetryPort] = None
        self.__summarizer: Optional[SummarizationPort] = None
        self.__qualifier: Optional[IntentQualifierPort] = None
        self.__realignment: Optional[RealignmentPolicy] = None

        self.__path_manager = path_manager
        self.__config: FathomConfiguration = FathomConfiguration()
        # Pre-bound application-layer config translator. When the
        # caller passes one via :meth:`with_runtime_configuration`, it rides
        # into :class:`FathomRunner` and on into the strategy so the
        # OCR / vision-localizer factories observe the same
        # :class:`FathomSettings` the caller built. Without this, the
        # strategy would fabricate ``RuntimeConfigLoader()`` itself,
        # which silently reads only fathom-prefixed env aliases and
        # misses deployment-prefixed names like
        # ``DRIZZ_GOOGLE_APPLICATION_CREDENTIALS_JSON``.
        #
        # Note: the raw :class:`FathomSettings` (Infrastructure layer)
        # never crosses this seam. The loader is the Application-layer
        # abstraction; the caller (e.g. ``FathomActivities``) is
        # responsible for binding settings to it before handing it in.
        # This keeps SA credentials and other secrets confined to the
        # caller's scope — they never become reachable from runner /
        # strategy code that might otherwise log them or pass them as
        # Temporal activity arguments.
        self.__runtime_configuration: Optional[RuntimeConfigLoader] = None

        # Set by the optional .with_assembly() path to build a dedicated qualifier LLM.
        self.__llm_factory: Optional[LLMFactoryPort] = None
        self.__assembly: Optional[RunAssemblyBuilder] = None

    def with_runtime_configuration(self, loader: RuntimeConfigLoader) -> FathomBuilder:
        """
        Attach a pre-bound :class:`RuntimeConfigLoader` so its settings
        ride all the way down to :class:`AdapterAssembly` inside the
        strategy. Required for any deployment whose env-var names
        differ from fathom's own ``FATHOM_*`` /
        ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` aliases (e.g.
        a deployment using a ``DRIZZ_`` prefix).

        The loader is an Application-layer object. The caller (e.g.
        Temporal worker registry) constructs it as
        ``RuntimeConfigLoader(settings=settings)`` and the raw
        :class:`FathomSettings` never crosses this boundary — keeping
        SA credentials and other secrets confined to caller scope.
        """

        self.__runtime_configuration = loader
        return self

    def with_device(self, port: DevicePort) -> FathomBuilder:
        """Register the device port used to drive the target."""

        self.__device = port
        return self

    def with_llm(self, port: LLMPort) -> FathomBuilder:
        """Register the planner LLM port that drives step decisions."""

        self.__llm = port
        return self

    def with_perception(self, port: PerceptionPort) -> FathomBuilder:
        """Register the perception port that reads on-screen state."""

        self.__perception = port
        return self

    def with_memory(self, port: MemoryPort) -> FathomBuilder:
        """Register the memory port backing durable run knowledge."""

        self.__memory = port
        return self

    def with_interaction(self, port: InteractionPort) -> FathomBuilder:
        """Register the interaction-storage port for conversation records."""

        self.__interaction = port
        return self

    def with_knowledge(self, port: KnowledgePort) -> FathomBuilder:
        """Register the knowledge port holding learned screen data."""

        self.__knowledge = port
        return self

    def with_signal(self, port: SignalPort) -> FathomBuilder:
        """Register the signal port carrying pause, resume, and cancel."""

        self.__signal = port
        return self

    def with_storage(
        self,
        port: StoragePort,
        configuration: StorageConfiguration,
    ) -> FathomBuilder:
        """
        Configure storage port and propagate its configuration into ``__config``.
        """

        self.__storage = port
        self.__config.storage = configuration

        return self

    def with_telemetry(self, port: TelemetryPort) -> FathomBuilder:
        """Register the telemetry port for client-facing run events."""

        self.__telemetry = port
        return self

    def with_summarizer(self, port: SummarizationPort) -> FathomBuilder:
        """Register the summarization port for run digests."""

        self.__summarizer = port
        return self

    def with_config(self, configuration: FathomConfiguration) -> FathomBuilder:
        """Replace the full Fathom configuration."""

        self.__config = configuration
        return self

    def with_execution_config(self, configuration: ExecutionConfiguration) -> FathomBuilder:
        """Set the execution engine configuration."""

        self.__config.engine = configuration
        return self

    def with_intent_config(self, configuration: IntentConfiguration) -> FathomBuilder:
        """Set the intent strategy configuration."""

        self.__config.intent = configuration
        return self

    def with_exploration_config(self, configuration: ExplorationConfiguration) -> FathomBuilder:
        """Set the exploration strategy configuration."""

        self.__config.exploration = configuration
        return self

    def with_qualifier_config(self, configuration: QualifierConfiguration) -> FathomBuilder:
        """
        Configure the intent qualifier so gate thresholds and inference knobs from the
        request reach both the qualifier construction and the runner's gate decision.
        """

        self.__config.qualifier = configuration
        return self

    def with_qualifier(self, port: IntentQualifierPort) -> FathomBuilder:
        """Register a pre-built intent qualifier port, bypassing composition."""

        self.__qualifier = port
        return self

    def with_assembly(
        self,
        *,
        assembly: RunAssemblyBuilder,
        llm_factory: Optional[LLMFactoryPort] = None,
    ) -> FathomBuilder:
        """
        Wire the bits the qualifier composer needs to build a dedicated
        qualifier LLM (separate from the planner LLM passed to .with_llm).

        When supplied, build() resolves the qualifier's model / timeout /
        retries through `assembly.build_qualifier_model_configuration(...)`
        and constructs a fresh LLM via `llm_factory.create(...)` — so the
        eval-tuned defaults in QualifierConfiguration.inference actually
        take effect for direct SDK callers (Enricher, integration tests,
        notebooks). When omitted, the qualifier falls back to running on
        the caller-supplied planner LLM and inference.* settings are
        ignored — preserves existing behavior for callers that don't opt in.
        """

        self.__assembly = assembly
        self.__llm_factory = llm_factory or LLMFactory()
        return self

    def with_realignment(self, policy: RealignmentPolicy) -> FathomBuilder:
        """Register the realignment policy governing context re-evaluation."""

        self.__realignment = policy
        return self

    def build(self) -> FathomRunner:
        """
        Validate the configured ports, apply defaults, and construct the runner.

        Raises:
            ConfigurationError: If required ports (device, llm) are not configured
        """

        from fathom.runtime.runner import FathomRunner

        if not self.__path_manager:
            self.__path_manager = SharedPathManager(settings=FathomSettings())

        if not self.__device:
            raise ConfigurationError("Device port is required. Call .with_device() before .build()")

        if not self.__llm:
            raise ConfigurationError("LLM port is required. Call .with_llm() before .build()")

        if not self.__perception:
            raise ConfigurationError(
                "Perception port is required. Call .with_perception() before .build()"
            )

        if not self.__memory:
            ledger = Ledger(database_path=self.__path_manager.get_ledger_db_path())
            provider = SQLiteMemoryProvider(
                database_path=self.__path_manager.get_knowledge_db_path()
            )
            self.__memory = SQLiteMemory(ledger=ledger, provider=provider)

        if not self.__interaction:
            raise ConfigurationError(
                "Interaction port is required. Call .with_interaction() before .build(). "
                "The Temporal activity and CLI executor wire it via "
                "InteractionFactory; tests should construct an adapter explicitly."
            )

        if not self.__knowledge:
            self.__knowledge = SQLiteKnowledge(path_manager=self.__path_manager)

        if not self.__signal:
            self.__signal = NoopSignal()

        if not self.__storage:
            self.__storage = LocalStorage(path_manager=self.__path_manager)

        if not self.__telemetry:
            self.__telemetry = StructlogAdapter()

        if not self.__summarizer:
            self.__summarizer = LLMSummarizer(llm=self.__llm)

        owned_resources: List[LLMPort] = []
        if not self.__qualifier:
            self.__qualifier, owned_resources = self.__compose_qualifier()

        if not self.__realignment:
            self.__realignment = RealignmentPolicy()

        # Deterministic sibling of the planner for decomposition; runner owns its cleanup.
        architect = self.__llm.derive(overrides=LLMConfiguration(temperature=0.0, use_cache=False))
        owned_resources.append(architect)

        return FathomRunner(
            llm=self.__llm,
            architect=architect,
            device=self.__device,
            config=self.__config,
            memory=self.__memory,
            interaction=self.__interaction,
            signal=self.__signal,
            storage=self.__storage,
            knowledge=self.__knowledge,
            telemetry=self.__telemetry,
            qualifier=self.__qualifier,
            summarizer=self.__summarizer,
            perception=self.__perception,
            realignment=self.__realignment,
            path_manager=self.__path_manager,
            runtime_configuration=self.__runtime_configuration,
            owned_resources=owned_resources,
        )

    def __compose_qualifier(self) -> tuple[IntentQualifierPort, List[LLMPort]]:
        """
        Construct the qualifier port and any resources the runner must own.

        Two paths:
          - assembly supplied (.with_assembly): build a dedicated qualifier LLM
            via the assembly's qualifier-model configuration. The dedicated LLM
            is returned alongside the qualifier so the runner can clean it up.
            inference.{model, timeout, max_retries, ...} take effect here.
          - assembly NOT supplied: fall back to the planner LLM. inference.* is
            stored on the config but ignored — preserves the pre-with_assembly
            behavior for direct callers that haven't opted in.

        This is the sync equivalent of QualifierComposer.compose(). Diverges
        only in cleanup-on-failure: if IntentQualifierFactory.create raises
        AFTER the dedicated LLM is built, the LLM is not awaited-closed (the
        builder is a startup-time call; on construction failure the process
        typically dies and the OS reclaims connections). Temporal / CLI paths
        use the proper async QualifierComposer with full cleanup; this sync
        version exists so direct SDK callers don't need to make build()
        async.
        """

        # build() raises ConfigurationError before reaching here if self.__llm is None;
        # cast removes the Optional from mypy's perspective without runtime overhead.
        planner_llm = cast("LLMPort", self.__llm)

        if self.__assembly is None or self.__llm_factory is None:
            qualifier = IntentQualifierFactory.create(
                llm=planner_llm, configuration=self.__config.qualifier
            )
            return qualifier, []

        if not self.__config.qualifier.enabled:
            qualifier = IntentQualifierFactory.create(
                llm=planner_llm, configuration=self.__config.qualifier
            )
            return qualifier, []

        qualifier_llm = self.__llm_factory.create(
            configuration=self.__assembly.build_qualifier_model_configuration(
                configuration=self.__config.qualifier
            )
        )
        qualifier = IntentQualifierFactory.create(
            llm=qualifier_llm, configuration=self.__config.qualifier
        )
        return qualifier, [qualifier_llm]


class Fathom:
    """
    Main entry point for Fathom library.
    """

    @staticmethod
    def builder(path_manager: Optional[SharedPathManager] = None) -> FathomBuilder:
        """Create a new builder instance."""

        return FathomBuilder(path_manager=path_manager)
