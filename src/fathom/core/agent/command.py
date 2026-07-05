from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.constants.capability import CompletionMode, PayloadField, TargetRequirement
from fathom.core.capability.catalog import CommandCatalog
from fathom.core.exceptions import ToolValidationError
from fathom.schemas.capability import CommandProfile
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

    def validate(
        self, *, command: ToolCommand, directive: Optional[ActionType] = None
    ) -> AcceptedCommand:
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

        self.__validate_directive(command=command, directive=directive)

        return AcceptedCommand(action_type=command.action_type, payload=command.payload)

    def __validate_directive(
        self, *, command: ToolCommand, directive: Optional[ActionType]
    ) -> None:
        """
        Reject commands that cannot satisfy a mandatory active directive contract.
        """

        if directive is None:
            return

        directed = self.__catalog.profile(action_type=directive)
        if directed.completion is not CompletionMode.CAPTURE_VERIFIED:
            return

        emitted = self.__catalog.profile(action_type=command.action_type)
        if emitted.completion is CompletionMode.CAPTURE_VERIFIED:
            return

        raise self.__error(
            message=self.__directive_message(
                directive=directive,
                required=self.__required_fields(profile=directed),
                action_type=command.action_type,
            )
        )

    @staticmethod
    def __directive_message(
        *, directive: ActionType, required: str, action_type: ActionType
    ) -> str:
        """
        Return model feedback for a command that cannot satisfy the active directive.
        """

        return (
            f"The active '{directive.value}' sub-goal needs an executable command with "
            f"{required}. The emitted '{action_type.value}' command cannot complete it."
        )

    @staticmethod
    def __required_fields(*, profile: CommandProfile) -> str:
        """
        Return the profile's required payload fields as model-readable text.
        """

        if not (required := profile.contract.required):
            return "no additional fields"

        return ", ".join(sorted(field.value.lower() for field in required))

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
