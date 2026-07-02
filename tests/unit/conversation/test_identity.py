from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from fathom.conversation.identity import InteractionIdentity


class InteractionIdentityTest(unittest.TestCase):
    """
    Verify deterministic id derivation across all entity kinds.
    """

    def setUp(self) -> None:
        """
        Bind a fresh identity helper for each test.
        """

        self.__identity = InteractionIdentity(execution="execution-1")

    def test_execution_property_returns_bound_execution(self) -> None:
        """
        Execution id is exposed as a read-only property.
        """

        self.assertEqual("execution-1", self.__identity.execution)

    def test_empty_execution_rejected_at_construction(self) -> None:
        """
        Identity helper refuses an empty execution id.
        """

        with self.assertRaises(ValueError):
            InteractionIdentity(execution="")

    def __assert_opaque_uuid(self, value: str) -> None:
        """
        Assert that an identifier is an opaque UUID string with no embedded scope text.
        """

        self.assertNotIn(":", value)
        self.assertEqual(value, str(uuid.UUID(value)))

    def test_root_task_id_is_stable(self) -> None:
        """
        Root task id is stable and opaque.
        """

        first = self.__identity.task()
        second = self.__identity.task()

        self.__assert_opaque_uuid(first)

        self.assertEqual(first, second)

    def test_step_task_id_carries_step_number_and_digest(self) -> None:
        """
        Step task ids are stable and change with the descriptor.
        """

        first = self.__identity.step_task(step_number=3, action_descriptor="tap login")
        second = self.__identity.step_task(step_number=3, action_descriptor="tap login")
        third = self.__identity.step_task(step_number=3, action_descriptor="tap signup")

        self.__assert_opaque_uuid(first)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_simple_message_id_is_stable_by_name(self) -> None:
        """
        Simple message ids depend only on the name qualifier.
        """

        result = self.__identity.message(name="result")
        request = self.__identity.message(name="request")

        self.__assert_opaque_uuid(request)
        self.__assert_opaque_uuid(result)

        self.assertNotEqual(request, result)
        self.assertEqual(request, self.__identity.message(name="request"))

    def test_derived_message_id_changes_with_qualifier(self) -> None:
        """
        Derived message ids change with the qualifier text.
        """

        first = self.__identity.derived_message(name="analysis", qualifier="alpha")
        second = self.__identity.derived_message(name="analysis", qualifier="alpha")
        third = self.__identity.derived_message(name="analysis", qualifier="beta")

        self.__assert_opaque_uuid(first)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_context_id_is_stable_by_name(self) -> None:
        """
        Context ids depend only on the name qualifier.
        """

        context = self.__identity.context(name="start")

        self.__assert_opaque_uuid(context)
        self.assertEqual(context, self.__identity.context(name="start"))

    def test_membership_id_is_stable(self) -> None:
        """
        Membership ids are stable without embedding thread, role, or actor text.
        """

        first = self.__identity.membership(thread="t-1", role="requester", actor="human-1")
        second = self.__identity.membership(thread="t-1", role="requester", actor="human-1")
        third = self.__identity.membership(thread="t-1", role="responder", actor="human-1")

        self.__assert_opaque_uuid(first)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_artifact_id_changes_with_path(self) -> None:
        """
        Artifact ids depend on the file path digest.
        """

        first = self.__identity.artifact(path=Path("/tmp/a.png"))
        second = self.__identity.artifact(path=Path("/tmp/a.png"))
        third = self.__identity.artifact(path=Path("/tmp/b.png"))

        self.__assert_opaque_uuid(first)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_job_id_is_stable_by_name(self) -> None:
        """
        Job ids depend only on the name qualifier.
        """

        job = self.__identity.job(name="memory")

        self.__assert_opaque_uuid(job)
        self.assertEqual(job, self.__identity.job(name="memory"))

    def test_script_id_changes_with_name(self) -> None:
        """
        Script ids depend on the logical script name.
        """

        first = self.__identity.script(name="final")
        second = self.__identity.script(name="final")
        third = self.__identity.script(name="step-010")

        self.__assert_opaque_uuid(first)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
