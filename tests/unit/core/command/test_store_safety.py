from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from unittest import TestCase
from unittest.mock import Mock

from fathom.constants import ActionType
from fathom.constants.tools import ToolName
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.capture.store import CaptureStore
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.action import ActionExecutor
from fathom.schemas.actions import Action
from fathom.schemas.capture import CaptureRequest
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import ExecutionResult


class StoreToolExposureTest(TestCase):
    """
    STORE is offered to the planner's execute_ui tool with its capture object.
    """

    @staticmethod
    def __action_schema() -> Dict[str, Any]:
        """
        Return the execute_ui 'action' parameter schema.
        """

        definitions = ToolRegistry.definitions(names=frozenset({ToolName.EXECUTE_UI}))
        declaration = definitions["function_declarations"][0]
        action_schema: Dict[str, Any] = declaration["parameters"]["properties"]["action"][
            "properties"
        ]
        return action_schema

    def test_store_present_in_execute_ui_action_enum(self) -> None:
        """
        The execute_ui action_type enum offers 'store'.
        """

        self.assertIn(ActionType.STORE.value, self.__action_schema()["action_type"]["enum"])

    def test_execute_ui_capture_object_requires_name_subject_value(self) -> None:
        """
        The execute_ui capture object exposes name/subject/value and requires all three.
        """

        capture = self.__action_schema()["capture"]

        self.assertIn("name", capture["properties"])
        self.assertIn("subject", capture["properties"])
        self.assertIn("value", capture["properties"])
        self.assertEqual(capture["required"], ["name", "subject", "value"])


class StoreExecutionTest(TestCase):
    """
    Pins deterministic STORE execution: semantic capture stores the supplied value.
    """

    __STORE_PROBE: str = "_ActionExecutor__execute_store"

    def __executor(self, *, store: CaptureStore) -> ActionExecutor:
        """
        Build an executor bound to a known capture store for inspection.
        """

        return ActionExecutor(
            device=Mock(),
            telemetry=Mock(),
            path_manager=Mock(),
            max_retries=0,
            catalog=CommandCatalogProvider().build(),
            capture_store=store,
        )

    @staticmethod
    def __action(*, value: Optional[str], subject: str = "price") -> Action:
        """
        Build a STORE action; value=None omits the capture request entirely.
        """

        capture = (
            None
            if value is None
            else CaptureRequest.model_construct(name="abc", subject=subject, value=value)
        )
        return Action(
            action_type=ActionType.STORE,
            rationale="capture a value",
            capture=capture,
        )

    def __run(
        self,
        *,
        executor: ActionExecutor,
        action: Action,
        observation: Optional[ScreenObservation],
    ) -> ExecutionResult:
        """
        Invoke the private STORE handler and return its execution result.
        """

        store_handler = getattr(executor, self.__STORE_PROBE)
        result: Tuple[ExecutionResult, Any] = store_handler(
            action=action,
            step_number=1,
            start_time=0.0,
        )
        return result[0]

    def test_value_store_succeeds(self) -> None:
        """
        STORE stores the supplied semantic value under the variable name.
        """

        store = CaptureStore()
        result = self.__run(
            executor=self.__executor(store=store),
            action=self.__action(value="₹499", subject="price"),
            observation=None,
        )

        self.assertTrue(result.success)
        self.assertEqual(store.read(name="abc").value, "₹499")

    def test_value_store_does_not_need_label_id_or_observation(self) -> None:
        """
        STORE is not an XML or manifest lookup.
        """

        store = CaptureStore()
        result = self.__run(
            executor=self.__executor(store=store),
            action=self.__action(value="₹86", subject="price of soap"),
            observation=None,
        )

        self.assertTrue(result.success)
        self.assertEqual(store.read(name="abc").value, "₹86")

    def test_missing_capture_request_fails_without_write(self) -> None:
        """
        A STORE with no capture request fails and writes nothing (no name to key on).
        """

        store = CaptureStore()
        result = self.__run(
            executor=self.__executor(store=store),
            action=self.__action(value=None),
            observation=None,
        )

        self.assertFalse(result.success)
        self.assertFalse(store.exists(name="abc"))

    def test_blank_value_fails_and_records_failed_capture(self) -> None:
        """
        Defensive execution rejects an empty value even if an invalid request reaches the executor.
        """

        store = CaptureStore()
        result = self.__run(
            executor=self.__executor(store=store),
            action=self.__action(value=" ", subject="price"),
            observation=None,
        )

        self.assertFalse(result.success)
        stored = store.read(name="abc")
        self.assertFalse(stored.success)
        self.assertIsNone(stored.value)
        self.assertIsNotNone(stored.reason)
