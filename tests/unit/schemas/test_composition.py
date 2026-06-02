from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from fathom.interfaces.lifecycle import RunnerLifecycle
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.schemas.composition import QualifierComposition, RunnerComposition
from fathom.schemas.qualification import QualificationVerdict


class _StubQualifier(IntentQualifierPort):
    """
    Minimal qualifier port double for composition shape tests.
    """

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Never called by these tests; raise so any accidental traffic is loud.
        """

        raise AssertionError("Composition shape tests must not invoke qualify()")


class QualifierCompositionTest(unittest.TestCase):
    """
    QualifierComposition carries the qualifier port and any owned resources.
    resources is a Tuple so frozen ownership cannot be subverted via append/extend.
    """

    def test_composition_holds_qualifier_and_resources(self) -> None:
        """
        Round-trip: qualifier port and owned resources land in the value object
        as a tuple.
        """

        qualifier = _StubQualifier()
        resource_one = MagicMock(spec=LLMPort)
        resource_two = MagicMock(spec=LLMPort)

        composition = QualifierComposition(
            qualifier=qualifier, resources=(resource_one, resource_two)
        )

        self.assertIs(composition.qualifier, qualifier)
        self.assertIsInstance(composition.resources, tuple)
        self.assertEqual(composition.resources, (resource_one, resource_two))

    def test_composition_resources_default_to_empty_tuple(self) -> None:
        """
        Permissive composition path defaults resources to an empty tuple.
        """

        composition = QualifierComposition(qualifier=_StubQualifier())
        self.assertEqual(composition.resources, ())
        self.assertIsInstance(composition.resources, tuple)

    def test_composition_field_reassignment_is_blocked(self) -> None:
        """
        frozen=True must block reassigning fields after construction.
        """

        composition = QualifierComposition(qualifier=_StubQualifier())
        with self.assertRaises(ValidationError):
            composition.qualifier = _StubQualifier()  # type: ignore[misc]

    def test_composition_resources_cannot_be_appended(self) -> None:
        """
        Regression: frozen=True alone does not prevent list mutation. Resources
        must be a tuple so callers cannot grow ownership after construction.
        """

        composition = QualifierComposition(
            qualifier=_StubQualifier(),
            resources=(MagicMock(spec=LLMPort),),
        )
        with self.assertRaises(AttributeError):
            composition.resources.append(MagicMock(spec=LLMPort))  # type: ignore[attr-defined]


class RunnerCompositionTest(unittest.TestCase):
    """
    RunnerComposition bundles the runner and any infrastructure resources owned
    by the composition root. resources is a Tuple for the same reason.
    """

    @staticmethod
    def __runner() -> MagicMock:
        """
        Build a runner stand-in that satisfies the RunnerLifecycle protocol
        (cleanup + cancel) so Pydantic validation accepts it.
        """

        return MagicMock(spec=RunnerLifecycle)

    def test_composition_holds_runner_and_resources(self) -> None:
        """
        Round-trip: runner reference and owned resources land in the value
        object as a tuple.
        """

        runner = self.__runner()
        resource = MagicMock(spec=LLMPort)
        composition = RunnerComposition(runner=runner, resources=(resource,))

        self.assertIs(composition.runner, runner)
        self.assertIsInstance(composition.resources, tuple)
        self.assertEqual(composition.resources, (resource,))

    def test_composition_resources_default_to_empty_tuple(self) -> None:
        """
        Exploration / disabled-qualifier paths have no owned resources.
        """

        composition = RunnerComposition(runner=self.__runner())
        self.assertEqual(composition.resources, ())
        self.assertIsInstance(composition.resources, tuple)

    def test_composition_field_reassignment_is_blocked(self) -> None:
        """
        frozen=True must block reassigning fields after construction.
        """

        composition = RunnerComposition(runner=self.__runner())
        with self.assertRaises(ValidationError):
            composition.resources = ()  # type: ignore[misc]

    def test_composition_resources_cannot_be_appended(self) -> None:
        """
        Regression: catches the case where switching back to List[LLMPort]
        would silently re-enable mutation. Tuple shape is load-bearing.
        """

        composition = RunnerComposition(
            runner=self.__runner(),
            resources=(MagicMock(spec=LLMPort),),
        )
        with self.assertRaises(AttributeError):
            composition.resources.append(MagicMock(spec=LLMPort))  # type: ignore[attr-defined]

    def test_runner_field_enforces_lifecycle_protocol(self) -> None:
        """
        Regression: RunnerComposition.runner must satisfy the RunnerLifecycle
        protocol (cleanup + cancel). A plain object without those methods is
        rejected at construction time, not at cleanup time.
        """

        class _NotALifecycle:
            """
            Bare object without cleanup() or cancel(); must fail validation.
            """

        with self.assertRaises(ValidationError):
            RunnerComposition(runner=_NotALifecycle())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
