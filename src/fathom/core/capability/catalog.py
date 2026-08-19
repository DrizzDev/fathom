from __future__ import annotations

from types import MappingProxyType
from typing import Dict, FrozenSet, Mapping

from fathom.constants import ActionType
from fathom.constants.capability import (
    COMMAND_REQUIREMENT_CHANNELS,
    NON_INTERACTIVE_CHANNELS,
    CompletionMode,
    ExecutionChannel,
    PayloadField,
    RecordMode,
    RetryMode,
    TargetRequirement,
)
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.capability import (
    CommandAvailabilityConfig,
    CommandContract,
    CommandProfile,
)
from fathom.schemas.requirement import CommandRequirement


class CommandCatalog:
    """
    Immutable registry mapping a command to its declared execution capability profile.
    """

    def __init__(self, *, profiles: Mapping[ActionType, CommandProfile]) -> None:
        """
        Bind an immutable command-to-profile mapping; subset catalogs (post-availability) are permitted.
        """

        self.__profiles: Mapping[ActionType, CommandProfile] = MappingProxyType(dict(profiles))

    def action_types(self) -> FrozenSet[ActionType]:
        """
        Return the commands this catalog knows about.
        """

        return frozenset(self.__profiles)

    def profile(self, *, action_type: ActionType) -> CommandProfile:
        """
        Return the profile for a command, failing fast when the command is unknown or disabled.
        """

        profile = self.__profiles.get(action_type)
        if profile is None:
            raise InvariantViolation(f"No command profile for '{action_type.value}'.")

        return profile

    def supports(self, *, action_type: ActionType) -> bool:
        """
        Return whether this catalog knows and enables the command.
        """

        return action_type in self.__profiles

    def admits_requirement(self, *, requirement: CommandRequirement) -> bool:
        """
        Return whether a canonical command requirement runs on an admissible execution channel.
        """

        return (
            self.profile(action_type=requirement.operation).channel in COMMAND_REQUIREMENT_CHANNELS
        )

    def is_spatial(self, *, action_type: ActionType) -> bool:
        """
        Return whether the command grounds to an on-screen target.
        """

        return self.profile(action_type=action_type).contract.target is not TargetRequirement.NONE

    def is_gesture(self, *, action_type: ActionType) -> bool:
        """
        Return whether the command grounds to a region rather than a single element.
        """

        return self.profile(action_type=action_type).contract.target is TargetRequirement.REGION

    def is_control(self, *, action_type: ActionType) -> bool:
        """
        Return whether the command runs on the control channel rather than the device.
        """

        return self.profile(action_type=action_type).channel is ExecutionChannel.CONTROL

    def is_device(self, *, action_type: ActionType) -> bool:
        """
        Return whether the command is a device command (everything outside the control channel).
        """

        return not self.is_control(action_type=action_type)

    def is_non_interactive(self, *, action_type: ActionType) -> bool:
        """
        Return whether the command completes without device interaction.
        """

        return self.profile(action_type=action_type).channel in NON_INTERACTIVE_CHANNELS

    def has_outer_retry(self, *, action_type: ActionType) -> bool:
        """
        Return whether the bounded outer retry loop wraps the command.
        """

        return self.profile(action_type=action_type).retry is RetryMode.OUTER


class CommandCatalogProvider:
    """
    Builds the full, exhaustive command catalog covering every known command.
    """

    def build(self) -> CommandCatalog:
        """
        Construct the full catalog, failing fast if any command is missing a profile.
        """

        profiles = self.__profiles()
        missing = set(ActionType) - set(profiles)
        if missing:
            names = ", ".join(sorted(str(action_type) for action_type in missing))
            raise InvariantViolation(f"Command catalog is missing profiles for: {names}.")

        return CommandCatalog(profiles=profiles)

    def __profiles(self) -> Dict[ActionType, CommandProfile]:
        """
        Declare the capability profile for every command.
        """

        element = CommandContract(target=TargetRequirement.ELEMENT)
        region = CommandContract(
            target=TargetRequirement.REGION, required=frozenset({PayloadField.SCROLL_TARGET})
        )
        typed = CommandContract(
            target=TargetRequirement.ELEMENT, required=frozenset({PayloadField.TEXT})
        )

        navigate = self.__profile(channel=ExecutionChannel.DEVICE)
        gesture = self.__profile(
            channel=ExecutionChannel.DEVICE, retry=RetryMode.INTERNAL, contract=region
        )
        memory = self.__profile(
            channel=ExecutionChannel.MEMORY, completion=CompletionMode.CLAIM_OR_TIMEOUT
        )

        return {
            ActionType.TAP: navigate.model_copy(update={"contract": element}),
            ActionType.TYPE: navigate.model_copy(update={"contract": typed}),
            ActionType.BACK: navigate,
            ActionType.HOME: navigate,
            ActionType.HIDE_KEYBOARD: navigate,
            ActionType.LONG_PRESS: navigate.model_copy(update={"contract": element}),
            ActionType.SWIPE: navigate.model_copy(update={"contract": region}),
            ActionType.SWIPE_UP: gesture,
            ActionType.SWIPE_DOWN: gesture,
            ActionType.SWIPE_LEFT: gesture,
            ActionType.SWIPE_RIGHT: gesture,
            ActionType.SCROLL: gesture,
            ActionType.WAIT: self.__profile(
                channel=ExecutionChannel.WAIT,
                completion=CompletionMode.CLAIM_OR_TIMEOUT,
                contract=CommandContract(required=frozenset({PayloadField.WAIT_SUBJECT})),
            ),
            ActionType.VALIDATE: self.__profile(
                channel=ExecutionChannel.OBSERVATION,
                completion=CompletionMode.CLAIM_VERIFIED,
                records=RecordMode.VALIDATION,
                contract=CommandContract(required=frozenset({PayloadField.SUBJECT})),
            ),
            ActionType.COMPLETE: self.__profile(
                channel=ExecutionChannel.TERMINAL, completion=CompletionMode.TERMINAL
            ),
            ActionType.SAVE_MEMORY: memory,
            ActionType.RETRIEVE_MEMORY: memory,
            ActionType.STORE: self.__profile(
                channel=ExecutionChannel.CAPTURE,
                completion=CompletionMode.CAPTURE_VERIFIED,
                records=RecordMode.CAPTURE,
                retry=RetryMode.NONE,
                contract=CommandContract(required=frozenset({PayloadField.CAPTURE})),
            ),
            ActionType.INFER: self.__profile(
                channel=ExecutionChannel.DEVICE, completion=CompletionMode.CLAIM_OR_TIMEOUT
            ),
            ActionType.UNKNOWN: navigate,
            ActionType.ASK_USER: self.__profile(
                channel=ExecutionChannel.CONTROL, completion=CompletionMode.CLAIM_OR_TIMEOUT
            ),
        }

    def __profile(
        self,
        *,
        channel: ExecutionChannel,
        completion: CompletionMode = CompletionMode.SCREEN_VERIFIED,
        records: RecordMode = RecordMode.ACTION,
        retry: RetryMode = RetryMode.OUTER,
        contract: CommandContract = CommandContract(),
    ) -> CommandProfile:
        """
        Build one command profile, defaulting to a screen-verified, outer-retried device command.
        """

        return CommandProfile(
            channel=channel,
            completion=completion,
            records=records,
            retry=retry,
            contract=contract,
        )


class CommandAvailabilityResolver:
    """
    Resolves the enabled command catalog from typed availability configuration.
    """

    def resolve(
        self, *, catalog: CommandCatalog, config: CommandAvailabilityConfig
    ) -> CommandCatalog:
        """
        Return a catalog with disabled commands removed so they are invisible to consumers.
        """

        enabled = {
            action_type: catalog.profile(action_type=action_type)
            for action_type in catalog.action_types()
            if action_type not in config.disabled
        }

        return CommandCatalog(profiles=enabled)
