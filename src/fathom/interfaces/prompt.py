"""Port definitions for prompt builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class NamedSection:
    """A provider-neutral prompt section.

    Shared contract type between ``core.prompts.*`` (which builds the
    sections) and adapter renderers (which format them for a specific
    provider). Lives in ``interfaces/`` so adapters can consume it
    without reaching into core internals.

    Attributes:
        name: Stable identifier for the section. Adapters use this to
            wrap the body in provider-specific markup (e.g.
            ``<NAME>...</NAME>`` for Gemini, ``## Name`` for markdown
            providers).
        body: The provider-neutral text body. Should never include
            adapter-specific tags.
        wrap: When True, adapters that use sectioned markup should wrap
            ``body`` in their section markup. When False, ``body`` is
            already self-contained prose and should be emitted as-is.
    """

    name: str
    body: str
    wrap: bool = True


@dataclass(frozen=True)
class SubGoalFocus:
    """Focus descriptor for the currently-active sub-goal.

    Passed to the prompt builder when sequential intent execution is
    active so that the adapter can render a single-sub-goal-focus section
    in its provider-specific format.
    """

    index: int
    total: int
    description: str


@dataclass(frozen=True)
class PromptUserContext:
    """Typed contract for the dynamic per-step prompt context.

    This is the explicit contract between the core vision service (which
    assembles the context) and the provider adapter (which renders it).
    It replaces the previous loose ``history=Any`` / ``memory=Dict`` /
    ``**kwargs`` signature on ``PromptBuilder.build_user_context``.

    Attributes:
        intent: The current user intent string.
        memory: Persistent cross-screen key/value memory. System keys
            (``context:``/``ctx_*``) should already be filtered out by
            the caller.
        trace: Ordered interaction history (most recent last). Each entry
            is a loose dict with at least ``action`` and ``observation``
            fields — adapters render this in a provider-specific way.
        milestones: High-level milestones achieved so far.
        guidance: Priority HITL guidance lines that must be respected by
            the agent. Rendered as a system override section.
        sub_goal_info: Current sub-goal focus descriptor when sequential
            execution is active. ``None`` when no sub-goal is in focus.
        screen_width: Live device viewport width in pixels, if known.
        screen_height: Live device viewport height in pixels, if known.
        use_xml: Whether XML grounding is enabled for this step.
        package_name: Target app package name when known (used for
            app-launch semantics).
        typing_text: Literal text the model should emit in a ``type``
            action, when the intent explicitly dictates it.
        current_screen_hash: Short hash of the live screen, used by the
            adapter to annotate stale trace observations.
        tracking_note: Loop-detection or cadence message that must be
            surfaced with maximum recency bias.
        loop_risk: True when the orchestrator has detected the agent is
            stuck in a repetitive loop. Adapters render a high-priority
            alert section telling the model to break out.
        failed_actions: Recently-failed action descriptions on the
            current screen. Adapters render a critical alert section
            telling the model NOT to retry these actions.
        last_action: Short description of the most recently-emitted
            action, surfaced to the model as recency context.
        delta_context: Optional structured screen-delta information
            (e.g. visible-anchor diffs since the previous step). Adapters
            decide how to serialize it.
    """

    intent: str
    memory: Mapping[str, str] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    milestones: Sequence[str] = field(default_factory=tuple)
    guidance: Sequence[str] = field(default_factory=tuple)
    sub_goal_info: Optional[SubGoalFocus] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    use_xml: bool = False
    package_name: Optional[str] = None
    typing_text: Optional[str] = None
    current_screen_hash: Optional[str] = None
    tracking_note: Optional[str] = None
    loop_risk: bool = False
    failed_actions: Sequence[str] = field(default_factory=tuple)
    last_action: Optional[str] = None
    delta_context: Optional[Mapping[str, Any]] = None


class PromptBuilder(ABC):
    """
    Abstract base class for building model-specific system prompts.
    """

    @abstractmethod
    def build(self) -> str:
        """
        Constructs the stable system instruction string (for caching).
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_context(self, context: PromptUserContext) -> str:
        """Constructs the dynamic user context string.

        Implementations must treat ``context`` as read-only and MUST NOT
        mutate any of its fields.
        """

        raise NotImplementedError


class ExportPromptBuilder(ABC):
    """
    Abstract builder for script-export prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for export generation.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(
        self,
        *,
        intent: str,
        goal_state: str,
        package_name: str,
        trace_payload: Sequence[Dict[str, Any]],
        action_catalog_lines: Sequence[str],
    ) -> str:
        """
        Build dynamic user prompt for export generation.
        """

        raise NotImplementedError


class DecompositionPromptBuilder(ABC):
    """
    Abstract builder for intent decomposition prompting.
    """

    @abstractmethod
    def build_system_instruction(self) -> str:
        """
        Build stable system instruction for decomposition generation.
        """

        raise NotImplementedError

    @abstractmethod
    def build_user_prompt(
        self,
        *,
        intent: str,
        stuck_sub_goal: Optional[str] = None,
        failure_reason: Optional[str] = None,
        suggested_next_action: Optional[str] = None,
        recent_actions: Sequence[str] = (),
    ) -> str:
        """Build the user prompt for decomposing an intent.

        When the caller is the replan path, ``stuck_sub_goal`` /
        ``failure_reason`` / ``suggested_next_action`` / ``recent_actions``
        carry the context that made the previous plan fail. Initial
        decomposition passes only ``intent``.
        """

        raise NotImplementedError
