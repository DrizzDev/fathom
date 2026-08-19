from __future__ import annotations

import ast
from typing import FrozenSet, List, Tuple

from governance.constants import RuleId
from governance.rules.base import Rule
from governance.schemas.finding import Violation
from governance.schemas.module import ParsedModule
from governance.schemas.selector import Selector


class PurityRule(Rule):
    """
    Enforces domain purity: pure-domain packages must not import logging, telemetry,
    or heavy-compute libraries. The pure-domain scope is injected from the taxonomy configuration, not hardcoded.
    """

    __FORBIDDEN_ROOTS: FrozenSet[str] = frozenset({"logging", "structlog", "cv2", "numpy", "PIL"})

    def __init__(self, *, domain: Tuple[str, ...]) -> None:
        """
        Bind the dotted package prefixes that must stay pure-domain.
        """

        self.__domain = domain

    @property
    def identifier(self) -> RuleId:
        """
        Stable identifier for this rule.
        """

        return RuleId.DOMAIN_PURITY

    @property
    def waivable(self) -> bool:
        """
        Domain-purity debt may be time-boxed while logging is migrated out of the domain.
        """

        return True

    def check(self, *, module: ParsedModule) -> List[Violation]:
        """
        Flag every forbidden import inside a pure-domain module.
        """

        if not self.__is_domain(package=module.package):
            return []

        violations: List[Violation] = []
        for node in ast.walk(module.tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for root in self.__roots(node=node):
                if root in self.__FORBIDDEN_ROOTS:
                    violations.append(self.__violation(module=module, line=node.lineno, root=root))
        return violations

    def __is_domain(self, *, package: str) -> bool:
        """
        Whether the package equals or is nested under an injected pure-domain prefix.
        """

        return any(
            package == prefix or package.startswith(f"{prefix}.") for prefix in self.__domain
        )

    @staticmethod
    def __roots(*, node: ast.AST) -> Tuple[str, ...]:
        """
        Return the top-level imported packages contributed by an import node.
        """

        if isinstance(node, ast.Import):
            return tuple(alias.name.split(".")[0] for alias in node.names)

        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            return (node.module.split(".")[0],)

        return ()

    def __violation(self, *, module: ParsedModule, line: int, root: str) -> Violation:
        """
        Build a domain-purity violation for a forbidden import.
        """

        return Violation(
            selector=Selector(rule=self.identifier, path=module.relative, detail=root),
            line=line,
            message=f"Pure-domain module '{module.package}' must not import '{root}'.",
        )
