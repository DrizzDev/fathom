from __future__ import annotations

from fathom.core.capability.binder import CommandBinder
from fathom.core.capability.catalog import CommandCatalog
from fathom.core.exceptions import TranslationError
from fathom.schemas.capture import CaptureIdentity
from fathom.schemas.proposal import (
    BoundCommand,
    CaptureProposal,
    CommandProposal,
    DecompositionProposal,
    ObservedProposal,
)
from fathom.schemas.success import CaptureSuccess, ObservationRequirement, ObservedSuccess, Success


class ProposalTranslator:
    """
    Translates an untrusted decomposition proposal into a canonical Success, or fails closed.
    """

    def __init__(self, *, catalog: CommandCatalog) -> None:
        """
        Compose the trusted command binder the translation delegates to.
        """

        self.__binder = CommandBinder(catalog=catalog)

    def translate(self, *, intent: str, proposal: DecompositionProposal) -> Success:
        """
        Translate one proposal into canonical Success, failing closed on an un-bindable command.
        """

        if isinstance(proposal, ObservedProposal):
            return ObservedSuccess(observation=ObservationRequirement(assertion=proposal.assertion))

        if isinstance(proposal, CaptureProposal):
            return CaptureSuccess(
                subject=proposal.subject,
                target=CaptureIdentity(name=proposal.name, provenance=proposal.provenance),
            )

        return self.__command(intent=intent, proposal=proposal)

    def __command(self, *, intent: str, proposal: CommandProposal) -> Success:
        """
        Bind a command proposal through the command binder or fail closed.
        """

        result = self.__binder.bind(intent=intent, proposal=proposal)

        if isinstance(result, BoundCommand):
            return result.success

        raise TranslationError(reason=result.reason.value)
