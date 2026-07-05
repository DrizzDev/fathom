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
    Covers catalog-backed validation of model-requested commands.
    """

    def test_accepts_store_with_capture_payload(self) -> None:
        """
        STORE is accepted only when the capture payload is present.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())
        command = gate.validate(
            command=ToolCommand(
                action_type=ActionType.STORE,
                payload=ExecuteAction(
                    action_type="store",
                    target_name="Price",
                    confidence=0.9,
                    capture=CaptureRequest(
                        name="item_price",
                        subject="price",
                        value="₹86",
                    ),
                ),
            )
        )

        self.assertEqual(command.action_type, ActionType.STORE)

    def test_rejects_store_without_capture_payload(self) -> None:
        """
        The catalog gate rejects STORE without the required capture field.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())
        payload = ExecuteAction.model_construct(action_type="store", confidence=0.9)

        with self.assertRaises(ToolValidationError):
            gate.validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.STORE,
                    payload=payload,
                )
            )

    def test_rejects_store_with_blank_capture_value(self) -> None:
        """
        The catalog gate rejects a structurally-present but blank capture value.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())
        payload = ExecuteAction.model_construct(
            action_type="store",
            confidence=0.9,
            capture=CaptureRequest.model_construct(name="item_price", subject="price", value=" "),
        )

        with self.assertRaises(ToolValidationError):
            gate.validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.STORE,
                    payload=payload,
                )
            )

    def test_rejects_type_without_text_payload(self) -> None:
        """
        TYPE cannot pass the catalog gate without text.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())
        payload = ExecuteAction.model_construct(
            action_type="type",
            target_name="Search box",
            confidence=0.9,
        )

        with self.assertRaises(ToolValidationError):
            gate.validate(command=ToolCommand(action_type=ActionType.TYPE, payload=payload))

    def test_rejects_validate_without_validation_subject(self) -> None:
        """
        VALIDATE cannot pass the catalog gate using only UI anchor text.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())
        payload = ExecuteAction.model_construct(
            action_type="validate",
            target_name="Phone Number",
            export_target="Phone Number input field",
            confidence=0.9,
        )

        with self.assertRaises(ToolValidationError):
            gate.validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.VALIDATE,
                    payload=payload,
                )
            )

    def test_accepts_validate_with_validation_subject(self) -> None:
        """
        VALIDATE passes when the assertion subject is explicit.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())
        command = gate.validate(
            command=ToolCommand(
                action_type=ActionType.VALIDATE,
                payload=ExecuteAction(
                    action_type="validate",
                    validation_subject="Login screen",
                    confidence=0.9,
                ),
            )
        )

        self.assertEqual(command.action_type, ActionType.VALIDATE)

    def test_store_directive_rejects_non_capture_command(self) -> None:
        """
        A STORE-directed sub-goal can only be satisfied by a capture-verified command.
        """

        gate = CommandGate(catalog=CommandCatalogProvider().build())

        with self.assertRaises(ToolValidationError):
            gate.validate(
                directive=ActionType.STORE,
                command=ToolCommand(
                    action_type=ActionType.VALIDATE,
                    payload=ExecuteAction(
                        action_type="validate",
                        validation_subject="Product price is visible",
                        confidence=0.9,
                    ),
                ),
            )

    def test_rejects_command_missing_from_catalog(self) -> None:
        """
        Disabled or unavailable commands fail before action materialization.
        """

        full = CommandCatalogProvider().build()
        catalog = CommandCatalog(
            profiles={ActionType.TAP: full.profile(action_type=ActionType.TAP)}
        )
        gate = CommandGate(catalog=catalog)

        with self.assertRaises(ToolValidationError):
            gate.validate(
                command=ToolCommand.model_construct(
                    action_type=ActionType.STORE,
                    payload=ExecuteAction.model_construct(action_type="store", confidence=0.9),
                )
            )
