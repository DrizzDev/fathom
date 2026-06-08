from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final, FrozenSet, List, Tuple

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.core.agent.tools.scope import ToolScope
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.subgoal import SubGoalKind
from fathom.schemas.tools import ToolPolicyContext


class IntentCorpus:
    """
    Loads the production intent corpus harvested from logs and debug artifacts.
    """

    __FILE: Final[Path] = Path(__file__).resolve().parents[5] / "tests" / "fixtures" / "intents.txt"

    @classmethod
    def all_intents(cls) -> Tuple[str, ...]:
        """
        Return every non-empty intent string from the corpus fixture.
        """

        if not cls.__FILE.exists():
            return ()

        return tuple(
            line.strip()
            for line in cls.__FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


class ToolScopeCorpusReplayTest(unittest.TestCase):
    """
    For every intent in the production corpus, the framework must produce the
    expected tool sets for every sub-goal shape it could encounter.
    """

    __VERIFY_TOOLS: Final[FrozenSet[ToolName]] = frozenset(
        {ToolName.VERIFY_GOAL, ToolName.VALIDATE_STATE},
    )

    @classmethod
    def __compute(cls, *, kind: SubGoalKind, hitl: bool) -> FrozenSet[ToolName]:
        """
        Reproduce the planner mapping and compute the per-turn tool set.
        """

        modes: set[TurnMode] = set()

        if kind == SubGoalKind.VALIDATION:
            modes.add(TurnMode.VERIFY)

        return (
            ToolScope(policies=DEFAULT_TOOL_POLICIES)
            .compute(
                context=ToolPolicyContext(
                    modes=frozenset(modes),
                    capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl)),
                ),
            )
            .names
        )

    def test_corpus_is_non_empty(self) -> None:
        """
        The corpus file must be present and contain at least one intent.
        """

        corpus = IntentCorpus.all_intents()

        self.assertGreater(len(corpus), 0, "intent corpus is empty; rebuild from /logs and /debug")

    def test_action_sub_goal_never_exposes_verification_tools(self) -> None:
        """
        For every corpus intent, an ACTION sub-goal turn must hide verification tools.
        """

        failures: List[str] = []
        corpus = IntentCorpus.all_intents()

        for intent in corpus:
            for hitl in (False, True):
                tools = self.__compute(kind=SubGoalKind.ACTION, hitl=hitl)
                if tools & self.__VERIFY_TOOLS:
                    failures.append(f"intent={intent!r} hitl={hitl} tools={sorted(tools)}")

        self.assertFalse(
            failures,
            "ACTION sub-goal leaked verification tools:\n" + "\n".join(failures),
        )

    def test_validation_sub_goal_always_exposes_verification_tools(self) -> None:
        """
        For every corpus intent, a VALIDATION sub-goal turn must expose both verification tools.
        """

        failures: List[str] = []
        corpus = IntentCorpus.all_intents()

        for intent in corpus:
            for hitl in (False, True):
                tools = self.__compute(kind=SubGoalKind.VALIDATION, hitl=hitl)
                missing = self.__VERIFY_TOOLS - tools
                if missing:
                    failures.append(f"intent={intent!r} hitl={hitl} missing={sorted(missing)}")

        self.assertFalse(
            failures,
            "VALIDATION sub-goal missed verification tools:\n" + "\n".join(failures),
        )

    def test_execute_ui_is_always_present(self) -> None:
        """
        Liveness — every (intent, sub-goal kind, hitl) combination must include EXECUTE_UI.
        """

        failures: List[str] = []
        corpus = IntentCorpus.all_intents()

        for intent in corpus:
            for kind in (SubGoalKind.ACTION, SubGoalKind.VALIDATION):
                for hitl in (False, True):
                    tools = self.__compute(kind=kind, hitl=hitl)
                    if ToolName.EXECUTE_UI not in tools:
                        failures.append(f"intent={intent!r} kind={kind} hitl={hitl}")

        self.assertFalse(
            failures,
            "EXECUTE_UI invariant violated:\n" + "\n".join(failures),
        )

    def test_ask_user_tracks_hitl_capability(self) -> None:
        """
        ASK_USER must be present iff HITL is enabled, across every corpus intent.
        """

        failures: List[str] = []
        corpus = IntentCorpus.all_intents()

        for intent in corpus:
            for kind in (SubGoalKind.ACTION, SubGoalKind.VALIDATION):
                for hitl in (False, True):
                    tools = self.__compute(kind=kind, hitl=hitl)
                    if (ToolName.ASK_USER in tools) != hitl:
                        failures.append(f"intent={intent!r} kind={kind} hitl={hitl}")

        self.assertFalse(
            failures,
            "ASK_USER did not track HITL capability:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
