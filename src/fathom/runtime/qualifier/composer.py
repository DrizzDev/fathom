from __future__ import annotations

import asyncio
import inspect
from logging import getLogger
from typing import TYPE_CHECKING, Tuple

from fathom.constants.run import ExecutionMode
from fathom.core.services.qualifier.factory import IntentQualifierFactory
from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.schemas.composition import QualifierComposition
from fathom.schemas.configuration import LLMConfiguration, QualifierConfiguration
from fathom.schemas.run import RunRequest

if TYPE_CHECKING:
    from fathom.runtime.assembly import RunAssemblyBuilder

logger = getLogger(__name__)


class QualifierComposer:
    """
    Composes the qualifier port with a dedicated low-temperature LLM when enabled.

    Lives in the runtime layer because it constructs infrastructure (a fresh LLM via the factory) and tracks owned resources.
    The qualifier port itself stays in core — pure domain behavior — and never owns its LLM.
    """

    def __init__(self, *, assembly: "RunAssemblyBuilder", llm_factory: LLMFactoryPort) -> None:
        """
        Wire the composer with the assembly that owns settings and the LLM factory.
        """

        self.__assembly = assembly
        self.__llm_factory = llm_factory

    @staticmethod
    def should_compose(*, request: RunRequest) -> bool:
        """
        Decide whether the request warrants a dedicated composed qualifier.

        Intent runs whose qualifier configuration is enabled get a dedicated
        composed qualifier with its own low-temperature LLM. Exploration runs
        never qualify — there is no user intent to gate. Intent runs that
        explicitly disable the qualifier fall through to the builder default.
        """

        if request.objective.mode != ExecutionMode.INTENT:
            return False

        return request.interaction.qualifier_configuration.enabled

    async def compose(
        self, *, planner_llm: LLMPort, configuration: QualifierConfiguration
    ) -> QualifierComposition:
        """
        Return a QualifierComposition with the qualifier port and any owned resources.

        When the qualifier is enabled, a dedicated low-temperature LLM is built and added to the composition's resources.
        When disabled, the permissive qualifier is composed with no extra resources to track.
        """

        resources: Tuple[LLMPort, ...] = ()
        qualifier_llm = planner_llm

        if configuration.enabled:
            qualifier_llm = self.__construct_dedicated_llm(configuration=configuration)
            resources = (qualifier_llm,)

        try:
            qualifier = IntentQualifierFactory.create(
                llm=qualifier_llm, configuration=configuration
            )
            logger.info(
                "[QualifierComposer] Composed qualifier port",
                extra={
                    "resources": len(resources),
                    "impl": type(qualifier).__name__,
                    "enabled": configuration.enabled,
                    "model": qualifier_llm.model_name,
                    "temperature": configuration.inference.temperature,
                    "thinking_level": configuration.inference.thinking_level,
                },
            )
            return QualifierComposition(qualifier=qualifier, resources=resources)
        except (Exception, asyncio.CancelledError):
            # CancelledError is BaseException-derived in 3.8+, so `except Exception`
            # would let it slip past and leak a dedicated qualifier LLM if a future
            # refactor introduces an `await` inside this block. Catch it explicitly,
            # drain, then re-raise so cancellation still propagates.
            if resources:
                await self.__cleanup_unreturned_resources(resources=resources)
            raise

    @staticmethod
    async def __cleanup_unreturned_resources(*, resources: Tuple[LLMPort, ...]) -> None:
        """
        Close resources created inside compose() when ownership never reaches the caller.
        """

        for resource in resources:
            try:
                result = resource.cleanup()
                if inspect.isawaitable(result):
                    await result
            except Exception as exception:
                logger.warning(
                    "[QualifierComposer] Failed to cleanup unreturned qualifier resource",
                    extra={"reason": str(exception), "resource": type(resource).__name__},
                )

    def __construct_dedicated_llm(self, *, configuration: QualifierConfiguration) -> LLMPort:
        """
        Build the dedicated low-temperature LLM, logging construction failures before re-raising.
        """

        llm_configuration: LLMConfiguration = self.__assembly.build_qualifier_model_configuration(
            configuration=configuration
        )
        try:
            return self.__llm_factory.create(configuration=llm_configuration)
        except Exception as exception:
            logger.warning(
                "[QualifierComposer] Dedicated qualifier LLM construction failed",
                extra={
                    "reason": str(exception),
                    "model": llm_configuration.model,
                    "location": llm_configuration.location,
                    "has_api_key": llm_configuration.api_key is not None,
                    "has_credentials": llm_configuration.credentials is not None,
                },
            )
            raise
