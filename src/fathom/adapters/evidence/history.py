from __future__ import annotations

import json
import time
from logging import getLogger
from typing import TYPE_CHECKING, List, Tuple

from pydantic import ValidationError

from fathom.constants import StepEvent
from fathom.constants.flow import LaunchProvenance
from fathom.constants.generation import COMPLETION_ASSERTIONS_FILENAME
from fathom.core.exceptions import ScriptExportError
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.core.services.generation.distiller import Distiller
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.interfaces.evidence import EvidenceSource
from fathom.interfaces.paths import HistoryPaths
from fathom.schemas.flow import CompletionAssertion, Evidence, RunObjective
from fathom.schemas.generation import Distillation, NormalizedTrace
from fathom.schemas.steps import StepHistory, StepRecord

if TYPE_CHECKING:
    from pathlib import Path

logger = getLogger(__name__)


class HistoryEvidenceSource(EvidenceSource):
    """
    Reads an execution's ordered trace, distils and normalizes it, and assembles Evidence.
    """

    __FILENAME = "history__execution.json"

    def __init__(
        self,
        *,
        distiller: Distiller,
        path_manager: HistoryPaths,
        assembler: EvidenceAssembler,
        normalizer: RunTraceNormalizer,
    ) -> None:
        """
        Bind the path port, the distiller, the launch normalizer, and the evidence assembler.
        """

        self.__distiller = distiller
        self.__assembler = assembler
        self.__normalizer = normalizer
        self.__path_manager = path_manager

    async def read(self, *, execution_id: str, objective: RunObjective) -> Evidence:
        """
        Load the execution trace, distil thrash, normalize launches, and assemble Evidence.
        """

        started = time.perf_counter()

        logger.info(
            "script evidence read started",
            extra={
                "event": "script.evidence.read.started",
                "execution.id": execution_id,
                "script.package": objective.package,
                "script.intent_present": bool(objective.intent.strip()),
            },
        )

        try:
            records = self.__records(execution_id=execution_id)
        except ScriptExportError as exception:
            logger.warning(
                "script evidence read failed",
                extra={
                    "event": "script.evidence.read.failed",
                    "execution.id": execution_id,
                    "script.package": objective.package,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                    "duration.ms": self.__elapsed(started=started),
                },
            )
            raise

        distillation = self.__distiller.distill(records=records)
        self.__log_distilled(execution_id=execution_id, raw=len(records), distillation=distillation)

        trace = self.__normalizer.normalize(records=distillation.records)
        self.__log_normalized(execution_id=execution_id, trace=trace)

        evidence = self.__assembler.assemble(
            trace=trace,
            goal=objective.goal,
            intent=objective.intent,
            package=objective.package,
            reason=distillation.reason,
            partial=distillation.partial,
            discarded=distillation.discarded,
            assertions=self.__assertions(execution_id=execution_id),
        )
        self.__log_assembled(execution_id=execution_id, evidence=evidence, started=started)

        return evidence

    def __log_distilled(self, *, execution_id: str, raw: int, distillation: Distillation) -> None:
        """
        Record the distillation outcome: input, kept, and dropped record counts.
        """

        logger.info(
            "script evidence distilled",
            extra={
                "event": "script.evidence.distilled",
                "script.input_count": raw,
                "execution.id": execution_id,
                "script.reason": distillation.reason,
                "script.partial": distillation.partial,
                "script.kept_count": len(distillation.records),
                "script.discarded_count": len(distillation.discarded),
            },
        )

    def __log_normalized(self, *, execution_id: str, trace: NormalizedTrace) -> None:
        """
        Record the launch-normalization outcome: launch markers by provenance and kept records.
        """

        launches = [entry.launch for entry in trace.entries if entry.launch is not None]
        transitions = sum(
            1 for launch in launches if launch.provenance is LaunchProvenance.LAUNCHER_TRANSITION
        )

        logger.info(
            "script evidence normalized",
            extra={
                "event": "script.evidence.normalized",
                "execution.id": execution_id,
                "script.launch_count": len(launches),
                "script.launcher_transition_count": transitions,
                "script.warm_start_count": len(launches) - transitions,
                "script.kept_record_count": sum(
                    1 for entry in trace.entries if entry.record is not None
                ),
            },
        )

    def __log_assembled(self, *, execution_id: str, evidence: Evidence, started: float) -> None:
        """
        Record the assembled evidence shape: steps, launches, validations, and captures.
        """

        logger.info(
            "script evidence assembled",
            extra={
                "event": "script.evidence.assembled",
                "execution.id": execution_id,
                "script.partial": evidence.partial,
                "script.step_count": len(evidence.steps),
                "duration.ms": self.__elapsed(started=started),
                "script.discarded_count": len(evidence.discarded),
                "script.launch_count": sum(1 for step in evidence.steps if step.launch is not None),
                "script.capture_count": sum(
                    1 for step in evidence.steps if step.capture is not None
                ),
                "script.validation_count": sum(
                    1 for step in evidence.steps if step.event == StepEvent.VALIDATION
                ),
                "script.assertion_count": len(evidence.assertions),
            },
        )

    @staticmethod
    def __elapsed(*, started: float) -> float:
        """
        Return milliseconds elapsed since the given perf-counter reading.
        """

        return round((time.perf_counter() - started) * 1000, 3)

    def __records(self, *, execution_id: str) -> List[StepRecord]:
        """
        Read and validate the execution's ordered trace, failing if absent or malformed.
        """

        path = self.__path(execution_id=execution_id)

        if not path.exists():
            raise ScriptExportError(f"No recorded execution trace for execution '{execution_id}'.")

        with path.open(mode="r") as handle:
            raw = json.load(handle)

        try:
            return list(StepHistory.model_validate(raw).history)
        except ValidationError as exception:
            raise ScriptExportError(
                f"Malformed execution trace for execution '{execution_id}': {exception}"
            ) from exception

    def __assertions(self, *, execution_id: str) -> Tuple[CompletionAssertion, ...]:
        """
        Load terminal verifier assertions for the execution when available.
        """

        path = self.__path_manager.get_history_directory(session_id=execution_id).joinpath(
            COMPLETION_ASSERTIONS_FILENAME
        )
        if not path.exists():
            return ()

        try:
            with path.open(mode="r") as handle:
                raw = json.load(handle)
            if not isinstance(raw, list):
                raise TypeError("completion assertions must be a JSON list")
            return tuple(CompletionAssertion.model_validate(item) for item in raw)
        except (json.JSONDecodeError, TypeError, ValidationError) as exception:
            raise ScriptExportError(
                f"Malformed completion assertions for execution '{execution_id}': {exception}"
            ) from exception

    def __path(self, *, execution_id: str) -> Path:
        """
        Resolve the execution trace path.
        """

        directory = self.__path_manager.get_history_directory(session_id=execution_id)
        return directory / self.__FILENAME
