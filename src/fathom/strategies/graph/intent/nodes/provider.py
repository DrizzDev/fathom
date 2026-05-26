from __future__ import annotations

import logging

from fathom.core.services.comparator import ScreenComparator
from fathom.core.services.criterion import CriterionChecker
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator
from fathom.strategies.graph.intent.nodes.effect import PostAction
from fathom.strategies.graph.intent.nodes.gate import ActionGate
from fathom.strategies.graph.intent.nodes.hitl import Hitl
from fathom.strategies.graph.intent.nodes.observer import ScreenObserver
from fathom.strategies.graph.intent.nodes.persistence import GraphStatePersistence

logger = logging.getLogger(__name__)


class IntentNodeProvider:
    """
    Provides LangGraph nodes for intent execution.
    Encapsulates dependencies and shared private logic.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        screen_comparator: ScreenComparator,
    ) -> None:
        """
        Initialize provider with shared context.
        """

        self.__context = context
        self.__screen_comparator = screen_comparator
        self.__persistence = GraphStatePersistence(context=context)
        self.__observer = ScreenObserver(context=context)
        self.__gate = ActionGate(context=context)
        self.__effects = PostAction(
            context=context,
            observer=self.__observer,
            comparator=screen_comparator,
        )
        self.__hitl = Hitl(context=context)
        self.__criterion_checker = CriterionChecker(llm=context.llm)
        self.__completion = SubGoalEvaluator(
            context=context,
            criterion_checker=self.__criterion_checker,
        )

    @property
    def context(self) -> GraphContext:
        """
        Return the shared graph context.
        """

        return self.__context

    @property
    def gate(self) -> ActionGate:
        """
        Return the action gate (localization helper).
        """

        return self.__gate

    @property
    def hitl(self) -> Hitl:
        """
        Return the HITL bridge.
        """

        return self.__hitl

    @property
    def observer(self) -> ScreenObserver:
        """
        Return the screen observer.
        """

        return self.__observer

    @property
    def effects(self) -> PostAction:
        """
        Return the post-action effect recorder.
        """

        return self.__effects

    @property
    def completion(self) -> SubGoalEvaluator:
        """
        Return the sub-goal completion evaluator.
        """

        return self.__completion

    @property
    def persistence(self) -> GraphStatePersistence:
        """
        Return the graph-state persistence helper.
        """

        return self.__persistence

    @property
    def screen_comparator(self) -> ScreenComparator:
        """
        Return the screen comparator service.
        """

        return self.__screen_comparator

    async def is_cancelled(self) -> bool:
        """
        Consolidated check for execution cancellation.
        """

        if self.__context.is_cancelled:
            return True

        signal = await self.__context.hitl.check_signal()
        if signal == "CANCELLED":
            self.__context.cancel()
            return True

        return False
