from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Tuple

from governance.constants import RuleId
from governance.rules.purity import PurityRule
from governance.rules.structure import DataclassRule
from governance.schemas.module import ParsedModule


class PurityRuleTest(unittest.TestCase):
    """
    Pins PurityRule against the injected pure-domain scope.
    """

    __DOMAIN: Tuple[str, ...] = ("fathom.schemas", "fathom.constants", "fathom.core.agent")

    @staticmethod
    def __module(*, package: str, code: str) -> ParsedModule:
        """
        Build a parsed module for a given dotted package from source text.
        """

        return ParsedModule(
            path=Path(f"{package}.py"),
            relative=f"src/{package.replace('.', '/')}.py",
            package=package,
            tree=ast.parse(code),
        )

    def test_forbidden_plain_import_in_domain_is_flagged(self) -> None:
        """
        A plain ``import logging`` inside a pure-domain module is a violation.
        """

        module = self.__module(package="fathom.schemas.configuration", code="import logging\n")
        violations = PurityRule(domain=self.__DOMAIN).check(module=module)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].selector.rule, RuleId.DOMAIN_PURITY)
        self.assertEqual(violations[0].selector.detail, "logging")

    def test_forbidden_from_import_in_domain_is_flagged(self) -> None:
        """
        A ``from numpy import ...`` inside a pure-domain module is a violation.
        """

        module = self.__module(package="fathom.core.agent.state", code="from numpy import array\n")
        violations = PurityRule(domain=self.__DOMAIN).check(module=module)

        self.assertEqual([violation.selector.detail for violation in violations], ["numpy"])

    def test_clean_domain_module_passes(self) -> None:
        """
        A domain module importing only permitted packages produces no violation.
        """

        module = self.__module(
            package="fathom.schemas.tasks", code="from fathom.constants import capability\n"
        )

        self.assertEqual(PurityRule(domain=self.__DOMAIN).check(module=module), [])

    def test_forbidden_import_outside_domain_is_ignored(self) -> None:
        """
        Forbidden imports outside the injected domain scope are not this rule's concern.
        """

        module = self.__module(package="fathom.adapters.vision", code="import numpy\n")

        self.assertEqual(PurityRule(domain=self.__DOMAIN).check(module=module), [])

    def test_sibling_package_sharing_a_prefix_is_ignored(self) -> None:
        """
        A prefix matches only on segment boundaries: fathom.core.agent_extra is not domain.
        """

        module = self.__module(package="fathom.core.agent_extra.thing", code="import logging\n")

        self.assertEqual(PurityRule(domain=self.__DOMAIN).check(module=module), [])

    def test_purity_rule_is_waivable(self) -> None:
        """
        Domain-purity debt may be time-boxed during migration.
        """

        self.assertTrue(PurityRule(domain=self.__DOMAIN).waivable)


class DataclassRuleTest(unittest.TestCase):
    """
    Pins DataclassRule including import-alias resolution.
    """

    @staticmethod
    def __module(*, code: str) -> ParsedModule:
        """
        Build a parsed module from source text.
        """

        return ParsedModule(
            path=Path("sample.py"),
            relative="src/fathom/sample.py",
            package="fathom.sample",
            tree=ast.parse(code),
        )

    def test_direct_decorator_is_flagged(self) -> None:
        """
        ``from dataclasses import dataclass`` used as a decorator is a violation.
        """

        code = "from dataclasses import dataclass\n\n\n@dataclass\nclass Point:\n    x: int\n"
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_direct_aliased_decorator_is_flagged(self) -> None:
        """
        ``from dataclasses import dataclass as dc`` is resolved and flagged.
        """

        code = "from dataclasses import dataclass as dc\n\n\n@dc\nclass Point:\n    x: int\n"
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_qualified_aliased_call_decorator_is_flagged(self) -> None:
        """
        ``import dataclasses as d`` used as ``@d.dataclass(frozen=True)`` is resolved and flagged.
        """

        code = "import dataclasses as d\n\n\n@d.dataclass(frozen=True)\nclass Point:\n    x: int\n"
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_pydantic_dataclass_is_flagged(self) -> None:
        """
        The ban is implementation-agnostic: a Pydantic dataclass is flagged too.
        """

        code = (
            "from pydantic.dataclasses import dataclass\n\n\n@dataclass\nclass Point:\n    x: int\n"
        )
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_assignment_alias_decorator_is_flagged(self) -> None:
        """
        A decorator bound through an assignment alias is resolved and flagged.
        """

        code = "import dataclasses\n\n\nentity = dataclasses.dataclass\n\n\n@entity\nclass Point:\n    x: int\n"
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_unrelated_decorator_is_ignored(self) -> None:
        """
        A non-dataclass class decorator is not a violation.
        """

        code = "def keep(target: type) -> type:\n    return target\n\n\n@keep\nclass Point:\n    x: int\n"

        self.assertEqual(DataclassRule().check(module=self.__module(code=code)), [])

    def test_bare_locally_defined_dataclass_name_is_flagged(self) -> None:
        """
        The ban is literal: any ``@dataclass`` is flagged, even a locally-defined name.
        """

        code = "def dataclass(target: type) -> type:\n    return target\n\n\n@dataclass\nclass Point:\n    x: int\n"
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_assignment_alias_chain_is_flagged(self) -> None:
        """
        Assignment aliases are resolved to a fixed point: ``second = first = dataclasses.dataclass``.
        """

        code = (
            "import dataclasses\n\n\nfirst = dataclasses.dataclass\nsecond = first\n\n\n"
            "@second\nclass Point:\n    x: int\n"
        )
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_annotated_assignment_alias_is_flagged(self) -> None:
        """
        An annotated assignment alias is resolved and flagged.
        """

        code = (
            "import dataclasses\nfrom typing import Any\n\n\nentity: Any = dataclasses.dataclass\n\n\n"
            "@entity\nclass Point:\n    x: int\n"
        )
        violations = DataclassRule().check(module=self.__module(code=code))

        self.assertEqual([violation.selector.detail for violation in violations], ["Point"])

    def test_dataclass_rule_is_not_waivable(self) -> None:
        """
        The dataclass ban is absolute: no debt record may waive it.
        """

        self.assertFalse(DataclassRule().waivable)
