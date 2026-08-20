from __future__ import annotations

import json
import threading
from collections import deque
from logging import getLogger
from typing import Callable, ClassVar, Dict, Optional

from fathom.schemas.metrics import WorkflowOperationMetrics

logger = getLogger(__name__)


class WorkflowSignalState:
    """
    Thread-safe in-process mirror of a single workflow's signal state.

    Workflow signal handlers and the activity-side TemporalSignalAdapter share
    the same worker process. This class lets the adapter read signal state
    without issuing Temporal queries.

    Workflow signal handlers write via mark_* / enqueue_context.
    The adapter reads via properties and waits on threading.Condition.
    """

    def __init__(self, *, workflow_id: str) -> None:
        self.__workflow_id = workflow_id

        self.__paused: bool = False
        self.__cancelled: bool = False
        self.__contexts: deque[str] = deque()
        self.__condition = threading.Condition()
        self.__metrics = WorkflowOperationMetrics()

    @property
    def workflow_id(self) -> str:
        """
        Workflow ID this state belongs to.
        """

        return self.__workflow_id

    @property
    def metrics(self) -> WorkflowOperationMetrics:
        """
        Operation counters for this workflow.
        """

        return self.__metrics

    def mark_paused(self) -> None:
        """
        Set paused flag and notify waiters.
        """

        with self.__condition:
            self.__paused = True
            self.__metrics.signals_received += 1

            self.__condition.notify_all()

        logger.info(f"[signal-state] workflow={self.__workflow_id} event=mark_paused state=paused")

    def mark_resumed(self) -> None:
        """
        Clear paused flag and notify waiters.
        """

        with self.__condition:
            self.__paused = False
            self.__metrics.signals_received += 1

            self.__condition.notify_all()

        logger.info(
            f"[signal-state] workflow={self.__workflow_id} event=mark_resumed state=running"
        )

    def mark_cancelled(self) -> None:
        """
        Set cancelled, clear paused, and notify waiters.
        """

        with self.__condition:
            self.__paused = False
            self.__cancelled = True
            self.__metrics.signals_received += 1

            self.__condition.notify_all()

        logger.info(
            f"[signal-state] workflow={self.__workflow_id} event=mark_cancelled state=cancelled"
        )

    def enqueue_context(self, *, context: str) -> None:
        """
        Append injected context and notify waiters.
        """

        with self.__condition:
            self.__contexts.append(context)
            self.__metrics.signals_received += 1
            self.__metrics.context_injections += 1

            depth = len(self.__contexts)
            self.__condition.notify_all()

        logger.info(
            f"[signal-state] workflow={self.__workflow_id} event=context_enqueued "
            f"queue_depth={depth} context_length={len(context)}"
        )

    def dequeue_context(self) -> Optional[str]:
        """
        Pop and return the next context, or None if empty.
        """

        with self.__condition:
            if self.__contexts:
                context = self.__contexts.popleft()
                self.__metrics.context_consumptions += 1
                remaining = len(self.__contexts)

                logger.info(
                    f"[signal-state] workflow={self.__workflow_id} event=context_dequeued "
                    f"remaining={remaining} context_length={len(context)}"
                )
                return context

            return None

    @property
    def paused(self) -> bool:
        """
        Whether pause is currently requested.
        """

        with self.__condition:
            return self.__paused

    @property
    def cancelled(self) -> bool:
        """
        Whether cancellation has been requested.
        """

        with self.__condition:
            return self.__cancelled

    def peek_context(self) -> Optional[str]:
        """
        Return the next context without consuming it.
        """

        with self.__condition:
            return self.__contexts[0] if self.__contexts else None

    def has_context(self) -> bool:
        """
        Whether any injected context is queued.
        """

        with self.__condition:
            return len(self.__contexts) > 0

    def wait_until(self, *, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
        """
        Block the calling thread until predicate is satisfied or timeout expires.

        Returns:
            True if the predicate was met, False on timeout.
        """

        with self.__condition:
            self.__metrics.wait_cycles += 1
            return self.__condition.wait_for(predicate, timeout=timeout)


class SignalStateRegistry:
    """
    Process-scoped registry of WorkflowSignalState instances.

    Provides a shared class-level instance so both workflow signal handlers
    and activity-side adapters can access the same state without module-level globals.
    """

    __instance: ClassVar[Optional[SignalStateRegistry]] = None
    __creation_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__states: Dict[str, WorkflowSignalState] = {}

    @classmethod
    def shared(cls) -> SignalStateRegistry:
        """
        Return the process-wide registry instance.
        """

        if cls.__instance is not None:
            return cls.__instance

        with cls.__creation_lock:
            if cls.__instance is None:
                cls.__instance = cls()

            return cls.__instance

    def get(self, *, workflow_id: str) -> WorkflowSignalState:
        """
        Get or create the signal state mirror for a workflow.
        """

        with self.__lock:
            if workflow_id not in self.__states:
                self.__states[workflow_id] = WorkflowSignalState(
                    workflow_id=workflow_id,
                )
                logger.info(
                    f"[signal-registry] workflow={workflow_id} event=state_created "
                    f"active_workflows={len(self.__states)}"
                )

            return self.__states[workflow_id]

    def release(self, *, workflow_id: str) -> None:
        """
        Remove signal state when a workflow completes and log operation summary.
        """

        with self.__lock:
            state = self.__states.pop(workflow_id, None)
            remaining = len(self.__states)

        if state is not None:
            metrics = json.dumps(state.metrics.model_dump(), separators=(",", ":"))
            logger.info(
                f"[signal-registry] workflow={workflow_id} event=state_released "
                f"active_workflows={remaining} metrics={metrics}"
            )
