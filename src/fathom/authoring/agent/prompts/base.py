from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from fathom.schemas.authoring.packet import AuthoringPacket


class AuthoringPrompt(ABC):
    """
    Strategy for rendering one authoring task kind into an LLM prompt.
    """

    __COMMON_CONTRACT: Tuple[str, ...] = (
        "You author replayable automation, not a transcript of execution logs.",
        "Use only the supplied evidence, artifacts, draft, review, and dialect reference.",
        "Write the command sequence a user should replay, preserving what actually happened.",
        "Improve wording only when the evidence supports the improvement.",
        "Represent captured values, validation states, targets, waits, scrolling, and conditionals according to their command semantics.",
        "If evidence is insufficient, set the Flow partial flag instead of inventing commands, targets, values, or assertions.",
        "Return only a valid Flow object for the requested dialect.",
    )
    __METHOD: Tuple[str, ...] = (
        "Read task.intent and the task-specific evidence view before selecting commands.",
        "Use dialect.commands as the supported command reference; do not emit unsupported nodes.",
        "Use dialect.guide.scenarios as the few-shot examples for authoring judgment.",
        "Use dialect.guide.lexicon as advisory UI terminology when it helps make a command complete and replayable.",
        "Use task.evidence.run.baseline as the deterministic scaffold when present; improve wording and structure only when the surrounding evidence, drafts, assertions, and artifacts support it.",
        "Use target.anchors and target.structure as target truth; use target.claim only when target.claim.verified is true.",
        "Use action, target, guard, capture, launch, observation, rationale, artifacts, and review as evidence, not as text to copy mechanically.",
        "When screenshots are attached, inspect visible UI state, target identity, and validation evidence; when manifests or UI trees are attached, use them to disambiguate element role and structure.",
        "Artifacts are optional; if they are absent, author from the recorded execution data without pretending visual or manifest evidence was available.",
        "Choose exact targets when the evidence identifies a stable UI element; choose relative or dynamic targets when the action selected an item by order, query, filter, condition, or runtime context.",
        "Collapse repeated attempts when one episode shows several tries toward one user-level purpose.",
        "Keep separate commands when the evidence shows separate user-level purposes.",
        "For completed run authoring, end with a Validate node grounded in supplied completion assertions; cite the assertion id in assertion_ids.",
    )

    def system_instruction(self) -> str:
        """
        Return the shared authoring contract and the task-specific objective.
        """

        method = "\n".join(f"- {line}" for line in self.__METHOD)
        contract = "\n".join(f"- {line}" for line in self.__COMMON_CONTRACT)

        return "\n\n".join(
            (
                "# Identity\n" + self.role(),
                "# Contract\n" + contract,
                "# Method\n" + method,
                "# Examples\nUse the dialect guide scenarios and command examples as reusable patterns, not as literal text to copy.",
                "# Output\nReturn only the structured Flow requested by the configured schema.",
                "# Objective\n" + self.objective(),
            )
        )

    def user_prompt(self, *, packet: AuthoringPacket) -> str:
        """
        Render the typed packet as the user prompt.
        """

        return "\n\n".join(
            (
                "# Dialect Reference",
                "Use this language reference for every output node. The commands are mandatory; the examples and lexicon are authoring guidance.",
                "```json",
                packet.dialect.model_dump_json(exclude_none=True),
                "```",
                "# Evidence",
                "Use this task evidence, drafts, review feedback, and artifact references as the only source of truth.",
                "```json",
                packet.task.model_dump_json(exclude_none=True),
                "```",
                "# Task",
                "Based on the dialect reference and evidence above, return the best Flow for this task.",
            )
        )

    @staticmethod
    def role() -> str:
        """
        Return the shared agent role.
        """

        return "You are Fathom's script authoring agent."

    @abstractmethod
    def objective(self) -> str:
        """
        Return the task-specific objective.
        """

        raise NotImplementedError
