from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final, FrozenSet, List, Tuple

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.registry import DEFAULT_TOOL_POLICIES
from fathom.core.agent.tools.scope import ToolScope
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
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
    Across the production intent corpus, the mode gate is intent-independent: an active goal keeps
    base UI tactics without VERIFY-only tools; the final-verification phase exposes them.
    """

    __VERIFY_TOOLS: Final[FrozenSet[ToolName]] = frozenset(
        {ToolName.VERIFY_GOAL, ToolName.VALIDATE_STATE},
    )

    @classmethod
    def __compute(cls, *, modes: FrozenSet[TurnMode], hitl: bool) -> FrozenSet[ToolName]:
        """
        Compute the per-turn tool set for the given mode set and HITL capability.
        """

        return (
            ToolScope(policies=DEFAULT_TOOL_POLICIES)
            .compute(
                context=ToolPolicyContext(
                    modes=modes,
                    capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=hitl)),
                ),
            )
            .names
        )

    def test_corpus_is_non_empty(self) -> None:
        """
        The corpus file must be present and contain at least one intent.
        """

        self.assertGreater(len(IntentCorpus.all_intents()), 0)

    def test_active_goal_never_exposes_verification_tools(self) -> None:
        """
        An active goal (empty mode set) hides verification tools for every corpus intent and HITL value.
        """

        failures: List[str] = []
        for intent in IntentCorpus.all_intents():
            for hitl in (False, True):
                tools = self.__compute(modes=frozenset(), hitl=hitl)
                if tools & self.__VERIFY_TOOLS:
                    failures.append(f"intent={intent!r} hitl={hitl} tools={sorted(tools)}")

        self.assertFalse(failures, "active goal leaked verification tools:\n" + "\n".join(failures))

    def test_final_verification_phase_exposes_verification_tools(self) -> None:
        """
        The final-verification phase (VERIFY mode) exposes both verification tools for every intent.
        """

        failures: List[str] = []
        for intent in IntentCorpus.all_intents():
            for hitl in (False, True):
                tools = self.__compute(modes=frozenset({TurnMode.VERIFY}), hitl=hitl)
                missing = self.__VERIFY_TOOLS - tools
                if missing:
                    failures.append(f"intent={intent!r} hitl={hitl} missing={sorted(missing)}")

        self.assertFalse(failures, "final phase missed verification tools:\n" + "\n".join(failures))

    def test_execute_ui_is_always_present(self) -> None:
        """
        Liveness — every (intent, mode, hitl) combination must include EXECUTE_UI.
        """

        failures: List[str] = []
        for intent in IntentCorpus.all_intents():
            for modes in (frozenset(), frozenset({TurnMode.VERIFY})):
                for hitl in (False, True):
                    if ToolName.EXECUTE_UI not in self.__compute(modes=modes, hitl=hitl):
                        failures.append(f"intent={intent!r} modes={modes} hitl={hitl}")

        self.assertFalse(failures, "EXECUTE_UI invariant violated:\n" + "\n".join(failures))

    def test_ask_user_tracks_hitl_capability(self) -> None:
        """
        ASK_USER must be present iff HITL is enabled, across every corpus intent.
        """

        failures: List[str] = []
        for intent in IntentCorpus.all_intents():
            for modes in (frozenset(), frozenset({TurnMode.VERIFY})):
                for hitl in (False, True):
                    tools = self.__compute(modes=modes, hitl=hitl)
                    if (ToolName.ASK_USER in tools) != hitl:
                        failures.append(f"intent={intent!r} modes={modes} hitl={hitl}")

        self.assertFalse(
            failures, "ASK_USER did not track HITL capability:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    unittest.main()
