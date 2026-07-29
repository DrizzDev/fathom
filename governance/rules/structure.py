from __future__ import annotations

import ast
from typing import FrozenSet, List, Set, Tuple

from governance.constants import RuleId
from governance.rules.base import Rule
from governance.schemas.finding import Violation
from governance.schemas.module import ParsedModule
from governance.schemas.selector import Selector


class DataclassRule(Rule):
    """
    Forbids any ``@dataclass`` decorator; Pydantic ``BaseModel`` is the required entity model
    (SKILL section 7). The ban is implementation-agnostic: standard-library and Pydantic
    dataclasses, aliased imports, and assignment aliases all resolve to the same violation.
    """

    __DECORATOR: str = "dataclass"

    @property
    def identifier(self) -> RuleId:
        """
        Stable identifier for this rule.
        """

        return RuleId.DATACLASS_FORBIDDEN

    @property
    def waivable(self) -> bool:
        """
        The dataclass ban is absolute: no debt record may waive it; violations must be converted.
        """

        return False

    def check(self, *, module: ParsedModule) -> List[Violation]:
        """
        Flag every class decorated with ``dataclass``, resolving import and assignment aliases first.
        """

        names = self.__aliases(tree=module.tree)

        violations: List[Violation] = []
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                if self.__is_dataclass(node=decorator, names=names):
                    violations.append(
                        self.__violation(module=module, line=decorator.lineno, name=node.name)
                    )
        return violations

    def __aliases(self, *, tree: ast.Module) -> FrozenSet[str]:
        """
        Local names that resolve to a ``dataclass`` decorator via import or assignment.
        """

        names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == self.__DECORATOR:
                        names.add(alias.asname or alias.name)

        assignments = self.__assignments(tree=tree)
        changed = True
        while changed:
            changed = False
            for target, value in assignments:
                if target not in names and self.__resolves(node=value, names=frozenset(names)):
                    names.add(target)
                    changed = True
        return frozenset(names)

    @staticmethod
    def __assignments(*, tree: ast.Module) -> List[Tuple[str, ast.expr]]:
        """
        Single-target name assignments and annotated assignments with a value.
        """

        pairs: List[Tuple[str, ast.expr]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    pairs.append((target.id, node.value))
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                pairs.append((node.target.id, node.value))
        return pairs

    def __is_dataclass(self, *, node: ast.expr, names: FrozenSet[str]) -> bool:
        """
        Whether a decorator expression resolves to a ``dataclass`` decorator.
        """

        return self.__resolves(node=node.func if isinstance(node, ast.Call) else node, names=names)

    def __resolves(self, *, node: ast.expr, names: FrozenSet[str]) -> bool:
        """
        Whether an expression's terminal symbol is ``dataclass``: the bare name (any provenance),
        a resolved alias, or a ``*.dataclass`` attribute.
        """

        if isinstance(node, ast.Name):
            return node.id == self.__DECORATOR or node.id in names

        if isinstance(node, ast.Attribute):
            return node.attr == self.__DECORATOR

        return False

    def __violation(self, *, module: ParsedModule, line: int, name: str) -> Violation:
        """
        Build a forbidden-dataclass violation for a decorated class.
        """

        return Violation(
            selector=Selector(rule=self.identifier, path=module.relative, detail=name),
            line=line,
            message=f"Class '{name}' uses @dataclass; use a Pydantic BaseModel instead.",
        )
