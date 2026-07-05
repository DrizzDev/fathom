from __future__ import annotations

from typing import Optional, Tuple

from fathom.constants.authoring import AuthoringKind
from fathom.constants.flow import IssueCode
from fathom.core.dialect.policy import Policy
from fathom.core.exceptions import LanguageComplianceError
from fathom.interfaces.dialect import Dialect
from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.flow import Evidence, Flow, Issue


class AuthoringReview:
    """
    Deterministic review result for an authored flow.
    """

    def __init__(self, *, text: str, issues: Tuple[Issue, ...]) -> None:
        """
        Bind review issues and rendered script text.
        """

        self.__text = text
        self.__issues = issues

    @property
    def issues(self) -> Tuple[Issue, ...]:
        """
        Return deterministic review issues.
        """

        return self.__issues

    @property
    def text(self) -> str:
        """
        Return rendered script text when render succeeded.
        """

        return self.__text

    @property
    def accepted(self) -> bool:
        """
        Return whether the reviewed flow can be published.
        """

        return not self.__issues and bool(self.__text)


class AuthoringReviewer:
    """
    Reviews authored flows with hard-truth policy and dialect checks.
    """

    def __init__(self, *, policy: Policy, dialect: Dialect) -> None:
        """
        Bind deterministic policy and target dialect.
        """

        self.__policy = policy
        self.__dialect = dialect

    def review(self, *, task: AuthoringTask, flow: Flow) -> AuthoringReview:
        """
        Render, syntax-check, and policy-review an authored flow.
        """

        issues: Tuple[Issue, ...] = ()
        evidence = self.__evidence(task=task)

        if task.kind is AuthoringKind.RUN and evidence is not None:
            issues = self.__policy.evaluate(flow=flow, evidence=evidence).issues

        try:
            text = self.__dialect.renderer.render(flow=flow).strip()
        except LanguageComplianceError as exception:
            return AuthoringReview(
                text="",
                issues=issues + (self.__render_issue(exception=exception),),
            )

        syntax = self.__dialect.checker.check(text=text)
        return AuthoringReview(issues=issues + syntax.issues, text=text)

    @staticmethod
    def __render_issue(*, exception: LanguageComplianceError) -> Issue:
        """
        Convert an unrenderable flow into deterministic review feedback.
        """

        return Issue(
            code=IssueCode.UNRENDERABLE_VALUE,
            message=f"Flow could not be rendered: {exception}",
        )

    @staticmethod
    def __evidence(*, task: AuthoringTask) -> Optional[Evidence]:
        """
        Return normalized execution evidence from the task view.
        """

        if task.evidence.run is not None:
            return task.evidence.run.source

        if task.evidence.step is not None:
            return task.evidence.step.source

        if task.evidence.repair is not None:
            return task.evidence.repair.source

        return None
