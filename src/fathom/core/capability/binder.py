from __future__ import annotations

from typing import Optional

from fathom.constants.command import CommandRejection
from fathom.core.capability.catalog import CommandCatalog
from fathom.schemas.proposal import BoundCommand, CommandBinding, CommandProposal, RejectedCommand
from fathom.schemas.success import CommandSuccess, SourceLocation, SourceSpan


class CommandBinder:
    """
    Admits an untrusted command proposal to a canonical CommandSuccess by catalog structure; the quote is diagnostic.
    """

    def __init__(self, *, catalog: CommandCatalog) -> None:
        """
        Bind to the command catalog whose admission authority the binder consults.
        """

        self.__catalog = catalog

    def bind(self, *, intent: str, proposal: CommandProposal) -> CommandBinding:
        """
        Admit the proposal on catalog structure, or return a typed rejection; the cited quote never gates.
        """

        operation = proposal.requirement.operation
        if not self.__catalog.supports(action_type=operation):
            return RejectedCommand(reason=CommandRejection.OPERATION_UNSUPPORTED)

        if not self.__catalog.admits_requirement(requirement=proposal.requirement):
            return RejectedCommand(reason=CommandRejection.CHANNEL_NOT_ADMITTED)

        return BoundCommand(
            success=CommandSuccess(
                requirement=proposal.requirement,
                source=self.__cite(intent=intent, quote=proposal.quote),
                postcondition=proposal.postcondition,
            )
        )

    @staticmethod
    def __cite(*, intent: str, quote: str) -> Optional[SourceSpan]:
        """
        Locate the cited quote as diagnostic provenance only; absence or ambiguity yields no citation, never a rejection.
        """

        start = intent.find(quote)
        if start < 0 or intent.find(quote, start + 1) >= 0:
            return None

        end = start + len(quote)
        return SourceSpan(quote=quote, location=SourceLocation(start=start, end=end))
