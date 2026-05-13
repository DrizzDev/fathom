from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, List, Optional, Sequence

from fathom.core.prompts.decomposition import DecompositionPromptBuilder
from fathom.core.prompts.factory import PromptFactory
from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.core.recovery.types import (
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryTrigger,
    ReplanOutcome,
)
from fathom.core.services.decomposer import DecompositionAugmentation, IntentDecomposer
from fathom.interfaces.llm import LLMPort, PromptPart

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext

logger = getLogger(__name__)


class _ReplanAugmentation(DecompositionAugmentation):
    """
    Replan-specific decomposer augmentation. Delegates prompt strings to
    the canonical :class:`DecompositionPromptBuilder` so all decomposition prompts live in the prompts layer.
    """

    def __init__(
        self,
        *,
        screenshot: bytes,
        trigger: RecoveryTrigger,
        stuck_sub_goal: str,
        failure_reason: str,
        recent_actions: List[str],
        builder: DecompositionPromptBuilder,
        suggested_next_action: Optional[str],
    ) -> None:
        self.__builder = builder
        self.__trigger = trigger
        self.__screenshot = screenshot
        self.__stuck_sub_goal = stuck_sub_goal
        self.__failure_reason = failure_reason
        self.__recent_actions = recent_actions
        self.__suggested_next_action = suggested_next_action

    def system_addendum(self) -> str:
        """
        Replan-specific system note: instructs the model to plan from the
        attached screenshot and treat failure context as paths to avoid.
        """

        return self.__builder.build_replan_system_note()

    def user_preamble(self) -> str:
        """
        Failure-evidence preamble prepended to the user prompt: trigger,
        stuck sub-goal, failure reason, suggested next action, recent
        actions. The trigger drives per-failure-mode framing so the
        decomposer can adapt instead of treating every replan identically.
        """

        return self.__builder.build_replan_user_preamble(
            trigger=self.__trigger,
            stuck_sub_goal=self.__stuck_sub_goal,
            failure_reason=self.__failure_reason,
            recent_actions=self.__recent_actions,
            suggested_next_action=self.__suggested_next_action,
        )

    def extra_prompt_parts(self) -> Sequence[PromptPart]:
        """
        Attach the current screenshot so the decomposer plans from the
        actual screen state instead of imagining the start screen.
        """

        return [self.__screenshot]


class ReplanRecovery(RecoveryStrategy):
    """
    Re-decomposes the remaining intent against the current screen when
    the agent is stuck on the active sub-goal.
    """

    def __init__(self, *, llm: LLMPort) -> None:
        self.__decomposer = IntentDecomposer(llm=llm)
        self.__prompt_builder = PromptFactory.get_decomposition_builder(model_name=llm.model_name)

    @classmethod
    def build(cls, context: "RecoveryContext") -> "ReplanRecovery":
        """
        Construct a :class:`ReplanRecovery` from the factory's recovery
        context, plucking only the ports this strategy requires.
        """

        return cls(llm=context.llm)

    @property
    def name(self) -> str:
        """
        Stable identifier used in ``RecoveryPolicy.strategies`` and logs.
        """

        return "replan"

    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Replan handles every stuck-evidence trigger. The trigger varies the
        decomposer's preamble framing (see :class:`_ReplanAugmentation`) so
        the same strategy can respond to different failure modes — loop,
        no-progress, unresolved target, budget exceeded, unactionable —
        without per-trigger code branches here.
        """

        return trigger in (
            RecoveryTrigger.ACTION_BLOCKED,
            RecoveryTrigger.VERIFY_REJECTED,
            RecoveryTrigger.LOOP_DETECTED,
            RecoveryTrigger.NO_PROGRESS,
            RecoveryTrigger.TARGET_UNRESOLVED,
            RecoveryTrigger.SUBGOAL_BUDGET_EXCEEDED,
            RecoveryTrigger.REPORT_UNACTIONABLE,
        )

    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Re-decompose the pending sub-goal tail against the current screen.

        Declines (returns ``None``) when there are no pending sub-goals
        left, the decomposer raises, or the decomposer produces an empty
        list — letting the coordinator fall through to the next strategy
        or the standard rejection path.
        """

        if not request.pending_sub_goals:
            logger.info(
                "[ReplanRecovery] no pending sub-goals to replace",
                extra={
                    "component": "replan",
                    "event": "decline_no_pending",
                    "trigger": request.trigger.value,
                    "stuck_sub_goal": request.stuck_sub_goal[:80],
                },
            )
            return None

        augmentation = _ReplanAugmentation(
            trigger=request.trigger,
            builder=self.__prompt_builder,
            failure_reason=request.reason,
            screenshot=request.capture.image,
            suggested_next_action=request.hint,
            stuck_sub_goal=request.stuck_sub_goal,
            recent_actions=list(request.recent_actions),
        )

        logger.info(
            "[ReplanRecovery] invoking decomposer pending=%d",
            len(request.pending_sub_goals),
            extra={
                "component": "replan",
                "event": "decomposer_invoke",
                "trigger": request.trigger.value,
                "hint_present": request.hint is not None,
                "screenshot_bytes": len(request.capture.image),
                "pending_count": len(request.pending_sub_goals),
                "recent_actions_count": len(request.recent_actions),
            },
        )

        try:
            new_sub_goals = await self.__decomposer.decompose(
                intent=". ".join(request.pending_sub_goals),
                augmentation=augmentation,
            )
        except Exception as exception:
            logger.warning(
                "[ReplanRecovery] decomposer failed: %s",
                exception,
                extra={
                    "component": "replan",
                    "error": str(exception),
                    "event": "decomposer_error",
                    "trigger": request.trigger.value,
                },
            )
            return None

        if not new_sub_goals:
            logger.warning(
                "[ReplanRecovery] decomposer returned no sub-goals",
                extra={"component": "replan", "event": "empty_decomposition"},
            )
            return None

        outcome = ReplanOutcome(
            new_sub_goals=new_sub_goals,
            summary=f"Replaced {len(request.pending_sub_goals)} sub-goal(s) with {len(new_sub_goals)}",
        )
        logger.info(
            "[ReplanRecovery] %s",
            outcome.summary,
            extra={
                "component": "replan",
                "event": "decomposer_success",
                "trigger": request.trigger.value,
                "new_count": len(new_sub_goals),
                "old_count": len(request.pending_sub_goals),
            },
        )
        return outcome
