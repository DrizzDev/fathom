from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Tuple

from governance.configuration import ConfigLoader
from governance.constants import ExpiryPolicy, RuleId
from governance.errors import GovernanceError
from governance.index import GitIndex, SourceIndex
from governance.reconcile import Reconciler
from governance.rules.base import Rule
from governance.rules.purity import PurityRule
from governance.rules.structure import DataclassRule
from governance.schemas.finding import Violation
from governance.schemas.module import ParsedModule
from governance.schemas.report import Audit
from governance.schemas.taxonomy import Taxonomy


class Checker:
    """
    Scans every first-party Python module in the repository and reconciles findings.

    First-party discovery is delegated to a source index (git by default), so scope follows the
    repository's own ignore rules rather than a directory-name allowlist.
    """

    __SOURCE_PREFIX = "src"
    __MANIFEST = "governance/debt.json"
    __TAXONOMY = "governance/taxonomy.json"

    def __init__(self, *, rules: Sequence[Rule], index: SourceIndex) -> None:
        """
        Bind the rules to evaluate and the source index that enumerates governed modules.
        """

        self.__rules: Tuple[Rule, ...] = tuple(rules)
        self.__index = index

    @classmethod
    def standard(cls, *, taxonomy: Taxonomy, index: Optional[SourceIndex] = None) -> "Checker":
        """
        Build the checker with the canonical rule set, injecting the taxonomy and source index.
        """

        return cls(
            rules=(PurityRule(domain=taxonomy.domain), DataclassRule()),
            index=index if index is not None else GitIndex(),
        )

    @classmethod
    def audit(cls, *, root: Path, today: date) -> Audit:
        """
        Scan the repository and reconcile the findings against its governance configuration.
        """

        loader = ConfigLoader()
        manifest = loader.manifest(path=root / cls.__MANIFEST)
        taxonomy = loader.taxonomy(path=root / cls.__TAXONOMY)
        checker = cls.standard(taxonomy=taxonomy)
        findings = checker.scan(repo=root)
        report = Reconciler().reconcile(
            findings=findings,
            manifest=manifest,
            today=today,
            waivable=checker.waivable(),
            warning=ExpiryPolicy.WARNING.value,
        )
        return Audit(mode=manifest.mode, provisional=taxonomy.provisional, report=report)

    def waivable(self) -> FrozenSet[RuleId]:
        """
        The identifiers of rules whose violations a debt record may accept.
        """

        return frozenset(rule.identifier for rule in self.__rules if rule.waivable)

    def scan(self, *, repo: Path) -> List[Violation]:
        """
        Evaluate every rule over every first-party module the index reports.
        """

        findings: List[Violation] = []
        for path in self.__index.paths(repo=repo):
            module = self.__parse(repo=repo, path=path)
            for rule in self.__rules:
                findings.extend(rule.check(module=module))
        return findings

    def __parse(self, *, repo: Path, path: Path) -> ParsedModule:
        """
        Parse one module into its tree and dotted package, failing fast on a syntax error.
        """

        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source=text, filename=str(path))
        except SyntaxError as exception:
            raise GovernanceError(f"cannot parse {path}: {exception}") from exception

        return ParsedModule(
            path=path,
            relative=str(path.relative_to(repo)),
            package=self.__package(repo=repo, path=path),
            tree=tree,
        )

    @classmethod
    def __package(cls, *, repo: Path, path: Path) -> str:
        """
        Derive the dotted package name of a module, stripping the ``src`` layout prefix.
        """

        parts = path.relative_to(repo).with_suffix("").parts
        if parts and parts[0] == cls.__SOURCE_PREFIX:
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
