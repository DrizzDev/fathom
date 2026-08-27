"""Build the resolved model-role manifest for a run."""

from __future__ import annotations

from typing import Any, List, Optional

from fathom.constants.manifest import (
    SHARED_ARCHITECT_NOTE,
    SHARED_QUALIFIER_NOTE,
    UNDECLARED_BINDING_NOTE,
    UNREAD_CONFIGURATION_NOTE,
    ModelRole,
    RoleBinding,
)
from fathom.core.embedding.capability import EmbeddingCapability
from fathom.interfaces.llm import LLMPort, ModelBound
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.interfaces.summarization import SummarizationPort
from fathom.schemas.abort import AbortDetectorConfiguration
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.llm import ModelIdentity
from fathom.schemas.localization import EnsembleMemberName
from fathom.schemas.manifest import RequestedModel, RoleResolution, RuntimeManifest
from fathom.schemas.perception import LocalizationEnsembleConfiguration


class ManifestResolver:
    """
    Projects the ports a runner was composed with into the resolved model-role manifest.

    Every role is read from the adapter instance that will serve it, so a role whose configured
    model composition never honors is reported the way it runs rather than the way it was asked for.
    """

    def __init__(
        self,
        *,
        planner: LLMPort,
        architect: LLMPort,
        qualifier: IntentQualifierPort,
        summarizer: SummarizationPort,
        configuration: FathomConfiguration,
    ) -> None:
        """
        Bind the composed ports and the configuration the run will execute under.
        """

        self.__planner = planner
        self.__architect = architect
        self.__qualifier = qualifier
        self.__summarizer = summarizer
        self.__configuration = configuration

    def resolve(
        self,
        *,
        steps: int,
        localization: LocalizationEnsembleConfiguration,
    ) -> RuntimeManifest:
        """
        Resolve every model-consuming role against the composed ports and the run's step budget.
        """

        planner = self.__identity(llm=self.__planner)

        roles: List[RoleResolution] = [
            self.__planner_role(planner=planner),
            self.__architect_role(planner=planner),
            self.__qualifier_role(),
            self.__summarizer_role(),
            self.__abort_role(planner=planner),
            self.__localizer_role(planner=planner, localization=localization),
            self.__embedder_role(planner=planner),
            self.__oracle_role(planner=planner),
        ]

        return RuntimeManifest(steps=steps, roles=tuple(roles))

    @staticmethod
    def __planner_role(*, planner: Optional[ModelIdentity]) -> RoleResolution:
        """
        The planner owns its adapter; the caller injects the port rather than requesting a model.
        """

        return RoleResolution(
            role=ModelRole.PLANNER,
            enabled=True,
            binding=RoleBinding.INDEPENDENT,
            resolved=planner,
        )

    def __architect_role(self, *, planner: Optional[ModelIdentity]) -> RoleResolution:
        """
        The architect is a derived sibling only when ``derive()`` returned a distinct adapter.
        """

        distinct = self.__architect is not self.__planner

        return RoleResolution(
            role=ModelRole.ARCHITECT,
            enabled=True,
            source=ModelRole.PLANNER,
            binding=RoleBinding.DERIVED if distinct else RoleBinding.SHARED,
            resolved=self.__identity(llm=self.__architect),
            note=None if distinct else SHARED_ARCHITECT_NOTE,
            requested=None if planner is None else RequestedModel(model=planner.model),
        )

    def __qualifier_role(self) -> RoleResolution:
        """
        The qualifier is independent only when composition built it a dedicated adapter.
        """

        configuration = self.__configuration.qualifier
        requested = RequestedModel(model=configuration.inference.model)

        if not configuration.enabled:
            return RoleResolution(
                role=ModelRole.QUALIFIER,
                enabled=False,
                binding=RoleBinding.NONE,
                requested=requested,
            )

        return self.__bound_role(
            role=ModelRole.QUALIFIER,
            port=self.__qualifier,
            requested=requested,
            shared_note=SHARED_QUALIFIER_NOTE,
        )

    def __summarizer_role(self) -> RoleResolution:
        """
        The summarizer reuses the planner adapter unless the caller supplied its own port.
        """

        return self.__bound_role(role=ModelRole.SUMMARIZER, port=self.__summarizer)

    def __abort_role(self, *, planner: Optional[ModelIdentity]) -> RoleResolution:
        """
        The abort detector always runs, always on the planner adapter, whatever it configures.
        """

        configuration = AbortDetectorConfiguration()

        return RoleResolution(
            role=ModelRole.ABORT,
            enabled=True,
            resolved=planner,
            source=ModelRole.PLANNER,
            binding=RoleBinding.SHARED,
            note=UNREAD_CONFIGURATION_NOTE,
            requested=RequestedModel(model=configuration.inference.model),
        )

    def __localizer_role(
        self,
        *,
        planner: Optional[ModelIdentity],
        localization: LocalizationEnsembleConfiguration,
    ) -> RoleResolution:
        """
        The vision localizer runs on the planner adapter when its ensemble member is configured.
        """

        enabled = localization.enabled and any(
            member in (EnsembleMemberName.VISION_LLM, EnsembleMemberName.GEMINI_VISION)
            for member in localization.members
        )

        return self.__shared_role(role=ModelRole.LOCALIZER, enabled=enabled, planner=planner)

    def __embedder_role(self, *, planner: Optional[ModelIdentity]) -> RoleResolution:
        """
        The embedder exists only when the planner adapter exposes a client it can share.
        """

        enabled = EmbeddingCapability.available(llm=self.__planner)

        return self.__shared_role(role=ModelRole.EMBEDDER, enabled=enabled, planner=planner)

    def __oracle_role(self, *, planner: Optional[ModelIdentity]) -> RoleResolution:
        """
        The shadow criterion oracle runs on the planner adapter when configuration enables it.
        """

        enabled = self.__configuration.oracle.enabled

        return self.__shared_role(role=ModelRole.ORACLE, enabled=enabled, planner=planner)

    def __bound_role(
        self,
        *,
        role: ModelRole,
        port: Any,
        requested: Optional[RequestedModel] = None,
        shared_note: Optional[str] = None,
    ) -> RoleResolution:
        """
        Build a role whose adapter is read from the port itself, comparing it against the planner.

        A port that does not expose its adapter is reported as undeclared rather than assumed to
        be independent, because assuming independence would hide a shared planner adapter.
        """

        adapter = self.__adapter(port=port)

        if adapter is None:
            return RoleResolution(
                role=role,
                enabled=True,
                requested=requested,
                binding=RoleBinding.UNKNOWN,
                note=UNDECLARED_BINDING_NOTE,
            )

        shared = adapter is self.__planner

        return RoleResolution(
            role=role,
            enabled=True,
            requested=requested,
            resolved=self.__identity(llm=adapter),
            binding=RoleBinding.SHARED if shared else RoleBinding.INDEPENDENT,
            source=ModelRole.PLANNER if shared else None,
            note=shared_note if shared else None,
        )

    @staticmethod
    def __shared_role(
        *,
        role: ModelRole,
        enabled: bool,
        planner: Optional[ModelIdentity],
    ) -> RoleResolution:
        """
        Build a role that has no adapter of its own and runs on the planner's when enabled.
        """

        return RoleResolution(
            role=role,
            enabled=enabled,
            resolved=planner if enabled else None,
            source=ModelRole.PLANNER if enabled else None,
            binding=RoleBinding.SHARED if enabled else RoleBinding.NONE,
        )

    @staticmethod
    def __adapter(*, port: Any) -> Optional[LLMPort]:
        """
        Read the adapter a port was composed with, or None when the port does not expose one.
        """

        if not isinstance(port, ModelBound):
            return None

        adapter = port.llm

        return adapter if isinstance(adapter, LLMPort) else None

    @staticmethod
    def __identity(*, llm: LLMPort) -> Optional[ModelIdentity]:
        """
        Read an adapter's declared identity, returning None when it declares nothing usable.

        Third-party adapters and test doubles reach this boundary, so one that answers with
        something other than a :class:`ModelIdentity` is reported as undeclared, never guessed.
        """

        identity = llm.identity

        return identity if isinstance(identity, ModelIdentity) else None
