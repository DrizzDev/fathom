from __future__ import annotations

from typing import Optional

from fathom.constants.capability import PayloadField, TargetRequirement
from fathom.core.capability.catalog import CommandCatalog
from fathom.core.exceptions import ToolValidationError
from fathom.schemas.gemini_tools import ExecuteAction
from fathom.schemas.results import ToolErrorFeedback
from fathom.schemas.tools import AcceptedCommand, ToolCommand


class CommandGate:
    """
    Validates model-requested commands against the command catalog.
    """

    def __init__(self, *, catalog: CommandCatalog) -> None:
        """
        Bind the catalog that defines executable command contracts.
        """

        self.__catalog = catalog

    def validate(self, *, command: ToolCommand) -> AcceptedCommand:
        """
        Return an accepted command or raise structured model feedback.
        """

        try:
            profile = self.__catalog.profile(action_type=command.action_type)
        except Exception as exception:
            raise self.__error(
                message=f"action_type='{command.action_type.value}' is not available."
            ) from exception

        for field in profile.contract.required:
            if not self.__has_field(payload=command.payload, field=field):
                raise self.__error(
                    message=(
                        f"action_type='{command.action_type.value}' is missing required "
                        f"payload field '{field.value}'."
                    )
                )

        if not self.__has_target(payload=command.payload, requirement=profile.contract.target):
            raise self.__error(
                message=(
                    f"action_type='{command.action_type.value}' does not satisfy target "
                    f"requirement '{profile.contract.target.value}'."
                )
            )

        return AcceptedCommand(action_type=command.action_type, payload=command.payload)

    @staticmethod
    def __has_field(*, payload: ExecuteAction, field: PayloadField) -> bool:
        """
        Return whether the payload satisfies one required command field.
        """

        if field is PayloadField.TEXT:
            return bool(payload.text or payload.text_to_type)

        if field is PayloadField.CAPTURE:
            capture = payload.capture
            return (
                capture is not None
                and bool(capture.name.strip())
                and bool(capture.value.strip())
                and bool(capture.subject.strip())
            )

        if field is PayloadField.SUBJECT:
            return bool(payload.validation_subject)

        if field is PayloadField.WAIT_SUBJECT:
            return bool(payload.wait_subject)

        if field is PayloadField.SCROLL_TARGET:
            return bool(payload.scroll_target)

        return False

    @classmethod
    def __has_target(cls, *, payload: ExecuteAction, requirement: TargetRequirement) -> bool:
        """
        Return whether the payload contains enough target grounding for the contract.
        """

        if requirement is TargetRequirement.NONE:
            return True

        if requirement is TargetRequirement.ELEMENT:
            return bool(cls.__target_text(payload=payload) or payload.label_id or payload.bbox)

        if requirement is TargetRequirement.REGION:
            return bool(
                payload.label_id or payload.bbox or payload.surface or payload.scroll_target
            )

        return False

    @staticmethod
    def __target_text(*, payload: ExecuteAction) -> Optional[str]:
        """
        Return the best textual target signal when present.
        """

        return (
            payload.target_name
            or payload.element_name
            or payload.export_target
            or payload.script_target
        )

    @staticmethod
    def __error(*, message: str) -> ToolValidationError:
        """
        Build structured feedback for command contract failures.
        """

        return ToolValidationError(
            ToolErrorFeedback(
                tool_name="execute_ui",
                tool_call_id=None,
                error_kind="validation",
                message=message,
            )
        )
