from __future__ import annotations

import inspect
import unittest

from fathom.core.services.qualifier import (
    LLMIntentQualifier,
    PermissiveIntentQualifier,
)
from fathom.interfaces.qualifier import IntentQualifierPort


class IntentQualifierPortContractTest(unittest.TestCase):
    """
    The qualifier port exposes exactly one method — qualify(). Lifecycle is
    the composition root's concern (RunnerComposition/QualifierComposition);
    putting cleanup() on the port forced wrappers and owns_llm flags. These
    tests lock in the binary port shape.
    """

    def test_port_exposes_only_qualify(self) -> None:
        """
        IntentQualifierPort must declare qualify() and nothing else as part of
        its public surface. New methods should require an explicit design
        review — this test fails loudly when one is added.
        """

        public_members = {
            name
            for name, _ in inspect.getmembers(IntentQualifierPort, predicate=inspect.isfunction)
            if not name.startswith("_")
        }
        self.assertEqual(public_members, {"qualify"})

    def test_llm_intent_qualifier_has_no_cleanup(self) -> None:
        """
        LLMIntentQualifier must not carry a cleanup() method; the composition
        root closes the dedicated LLM via RunnerComposition.resources instead.
        """

        self.assertFalse(hasattr(LLMIntentQualifier, "cleanup"))

    def test_permissive_qualifier_has_no_cleanup(self) -> None:
        """
        PermissiveIntentQualifier likewise must not carry cleanup() — it owns
        no infrastructure, so lifecycle should not appear on its surface.
        """

        self.assertFalse(hasattr(PermissiveIntentQualifier, "cleanup"))


if __name__ == "__main__":
    unittest.main()
