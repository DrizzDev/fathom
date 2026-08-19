from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Optional

from fathom.constants.turn.oracle import OracleThreshold
from fathom.interfaces.oracle import OraclePort
from fathom.schemas.criterion import CriterionVerdict, Verdict

logger = getLogger(__name__)


class OracleRecorder:
    """
    Shadow-reads the active criterion against the settled screen off the critical path; adds no step latency.
    """

    def __init__(self, *, oracle: Optional[OraclePort] = None) -> None:
        """
        Bind the optional oracle; a recorder without one is inert.
        """

        self.__oracle = oracle

    def observe(
        self,
        *,
        turn: int,
        workflow_id: str,
        image: Optional[bytes],
        criterion: Optional[str],
    ) -> None:
        """
        Fire one off-critical-path oracle reading; the step never waits on it.
        """

        if (oracle := self.__oracle) is None or not criterion or not image:
            return

        asyncio.create_task(
            self.__read(
                turn=turn,
                image=image,
                oracle=oracle,
                criterion=criterion,
                workflow_id=workflow_id,
            ),
            name="oracle.trial.reader",
        )

    @classmethod
    async def __read(
        cls,
        *,
        turn: int,
        image: bytes,
        criterion: str,
        workflow_id: str,
        oracle: OraclePort,
    ) -> None:
        """
        Read the verdict and log it; never raise, never consume live.
        """

        start = time.time()

        try:
            verdict = await oracle.read(criterion=criterion, image=image)
        except Exception as exception:
            logger.warning(
                "Oracle reading failed; nothing consumed",
                extra={
                    "event": "oracle.trial.failed",
                    "workflow.id": workflow_id,
                    "oracle.turn": turn,
                    "oracle.latency": round(time.time() - start, 3),
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return

        cls.__log(
            turn=turn, workflow_id=workflow_id, criterion=criterion, verdict=verdict, start=start
        )

    @staticmethod
    def __log(
        *,
        turn: int,
        workflow_id: str,
        criterion: str,
        verdict: Verdict,
        start: float,
    ) -> None:
        """
        Emit the shadow row; the logged proposal degrades to UNCLEAR below the confidence floor.
        """

        proposed = (
            verdict.outcome
            if verdict.confidence >= OracleThreshold.CONFIDENCE_FLOOR
            else CriterionVerdict.UNCLEAR
        )
        logger.info(
            "Oracle trial read",
            extra={
                "event": "oracle.trial.read",
                "workflow.id": workflow_id,
                "oracle.turn": turn,
                "oracle.criterion": criterion[:120],
                "oracle.proposed": proposed.value,
                "oracle.latency": round(time.time() - start, 3),
                "oracle.verdict": verdict.model_dump(mode="json"),
            },
        )
