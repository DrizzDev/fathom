from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.command import CommandGate
from fathom.core.capability.catalog import CommandCatalog, CommandCatalogProvider
from fathom.core.exceptions import ToolValidationError
from fathom.schemas.capture import CaptureRequest
from fathom.schemas.gemini_tools import ExecuteAction
from fathom.schemas.tools import ToolCommand


class CommandGateTest(unittest.TestCase):
    """
    Covers catalog-backed structural validation of model-requested commands.

    The gate validates command structure only (available action, required fields, target
    grounding). Requirement admission and completion authority live elsewhere, so a well-formed
    command is admitted regardless of the active goal — preparatory tactics are never gate-blocked.
    """

    @staticmethod
    def __gate() -> CommandGate:
        return CommandGate(catalog=CommandCatalogProvider().build())

    # ── Structural acceptance ─────────────────────────────────────────────

    def test_accepts_store_with_capture_payload(self) -> None:
        """
        STORE is accepted only when the capture payload is present.
        """

        command = self.__gate().validate(command=self.__store())
        self.assertEqual(command.action_type, ActionType.STORE)

    def test_accepts_validate_with_validation_subject(self) -> None:
        """
        VALIDATE passes when the assertion subject is explicit.
        """

        command = self.__gate().validate(command=self.__validate())
        self.assertEqual(command.action_type, ActionType.VALIDATE)

    # ── Structural rejection ──────────────────────────────────────────────

    def test_rejects_store_without_capture_payload(self) -> None:
        """
        The catalog gate rejects STORE without the required capture field.
        """

        payload = ExecuteAction.model_construct(action_type="store", confidence=0.9)
        with self.assertRaises(ToolValidationError):
            self.__gate().validate(
                command=ToolCommand.model_construct(action_type=ActionType.STORE, payload=payload)
            )

    def test_rejects_store_with_blank_capture_value(self) -> None:
        """
        The catalog gate rejects a structurally-present but blank capture value.
        """

        payload = ExecuteAction.model_construct(
            action_type="store",
            confidence=0.9,
            capture=CaptureRequest.model_construct(name="item_price", subject="price", value=" "),
        )
        with self.assertRaises(ToolValidationError):
            self.__gate().validate(
                command=ToolCommand.model_construct(action_type=ActionType.STORE, payload=payload)
            )

    def test_rejects_type_without_text_payload(self) -> None:
        """
        TYPE cannot pass the catalog gate without text.
        """

        payload = ExecuteAction.model_construct(
            action_type="type", target_name="Search box", export_target="Search box", confidence=0.9
        )
        with self.assertRaises(ToolValidationError):
            self.__gate().validate(command=ToolCommand(action_type=ActionType.TYPE, payload=payload))

    def test_rejects_validate_without_validation_subject(self) -> None:
        """
        VALIDATE cannot pass the catalog gate using only UI anchor text.
        """

        payload = ExecuteAction.model_construct(
            action_type="validate",
            target_name="Phone Number",
            export_target="Phone Number input field",
            confidence=0.9,
        )
        with self.assertRaises(ToolValidationError):
            self.__gate().validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.VALIDATE, payload=payload
                )
            )

    def test_rejects_validate_with_whitespace_validation_subject(self) -> None:
        """
        A whitespace-only subject is no assertion; the canonical constructor rejects it.
        """

        payload = ExecuteAction.model_construct(
            action_type="validate", validation_subject="   ", confidence=0.9
        )
        with self.assertRaises(ToolValidationError):
            self.__gate().validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.VALIDATE, payload=payload
                )
            )

    def test_rejects_command_missing_from_catalog(self) -> None:
        """
        Disabled or unavailable commands fail before action materialization.
        """

        full = CommandCatalogProvider().build()
        catalog = CommandCatalog(profiles={ActionType.TAP: full.profile(action_type=ActionType.TAP)})
        gate = CommandGate(catalog=catalog)

        with self.assertRaises(ToolValidationError):
            gate.validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.STORE,
                    payload=ExecuteAction.model_construct(action_type="store", confidence=0.9),
                )
            )

    # ── Negative regression: no directive-based admission ─────────────────

    def test_well_formed_commands_admitted_regardless_of_goal(self) -> None:
        """
        Regression: the gate performs no directive/completion-mode admission. A well-formed
        proof-bearing or preparatory command is admitted structurally — the old symmetric gate that
        rejected preparatory tactics under a mismatched goal must not recur.
        """

        gate = self.__gate()
        for command in (
            self.__store(),
            self.__validate(),
            self.__tap(),
            self.__type(),
            self.__scroll(),
            self.__complete(),
            self.__ask_user(),
            self.__save_memory(),
        ):
            with self.subTest(action_type=command.action_type.value):
                accepted = gate.validate(command=command)
                self.assertEqual(accepted.action_type, command.action_type)

    # ── Command builders ──────────────────────────────────────────────────

    @staticmethod
    def __store() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.STORE,
            payload=ExecuteAction(
                action_type="store",
                target_name="Price",
                confidence=0.9,
                capture=CaptureRequest(name="product_amount", subject="price", value="₹276"),
            ),
        )

    @staticmethod
    def __validate() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.VALIDATE,
            payload=ExecuteAction(
                action_type="validate",
                validation_subject="Customer rating is >= 4.2",
                confidence=0.9,
            ),
        )

    @staticmethod
    def __tap() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.TAP,
            payload=ExecuteAction(
                action_type="tap",
                target_name="Buy Now",
                export_target="Buy Now button",
                confidence=0.9,
            ),
        )

    @staticmethod
    def __type() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.TYPE,
            payload=ExecuteAction(
                action_type="type",
                target_name="Search field",
                export_target="search field",
                text_to_type="Ghar soaps",
                confidence=0.9,
            ),
        )

    @staticmethod
    def __scroll() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.SCROLL,
            payload=ExecuteAction(
                action_type="scroll", label_id="1", scroll_target="product list", confidence=0.9
            ),
        )

    @staticmethod
    def __complete() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.COMPLETE,
            payload=ExecuteAction(action_type="complete", confidence=0.9),
        )

    @staticmethod
    def __ask_user() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.ASK_USER,
            payload=ExecuteAction(
                action_type="ask_user", text="Which product should I select?", confidence=0.9
            ),
        )

    @staticmethod
    def __save_memory() -> ToolCommand:
        return ToolCommand(
            action_type=ActionType.SAVE_MEMORY,
            payload=ExecuteAction(action_type="save_memory", confidence=0.9),
        )


if __name__ == "__main__":
    unittest.main()
