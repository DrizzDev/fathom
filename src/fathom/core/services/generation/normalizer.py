from __future__ import annotations

from typing import List, Optional, Sequence

from fathom.constants.flow import LaunchProvenance
from fathom.interfaces.packages import PackageClassifier
from fathom.schemas.generation import LaunchMarker, NormalizedEntry, NormalizedTrace
from fathom.schemas.steps import StepRecord


class RunTraceNormalizer:
    """
    Normalizes a workflow trace into ordered steps with deterministic, typed launch markers.
    """

    def __init__(self, *, classifier: PackageClassifier) -> None:
        """
        Bind the package classifier used to recognise launcher-executed steps.
        """

        self.__classifier = classifier

    def normalize(self, *, records: Sequence[StepRecord]) -> NormalizedTrace:
        """
        Collapse launcher steps into typed launches and keep all real-app and system steps in order.
        """

        ordered = sorted(records, key=lambda record: record.step_number)

        entries: List[NormalizedEntry] = []
        current_app: Optional[str] = None
        pending_launcher: List[int] = []

        for record in ordered:
            execution = self.__package(activity=record.execution_activity)

            if execution is not None and self.__classifier.is_launcher(package=execution):
                pending_launcher.append(record.step_number)
                observed = self.__package(activity=record.activity)

                if (
                    observed
                    and not self.__classifier.is_launcher(package=observed)
                    and observed != current_app
                ):
                    entries.append(self.__transition(target=observed, steps=pending_launcher))
                    current_app = observed
                    pending_launcher = []

                continue

            if execution is None:
                entries.append(NormalizedEntry(record=record))
                pending_launcher = []
                continue

            if current_app is None:
                entries.append(self.__open(target=execution, steps=pending_launcher))
                current_app = execution
                pending_launcher = []
            elif pending_launcher and execution != current_app:
                entries.append(self.__transition(target=execution, steps=pending_launcher))
                current_app = execution
                pending_launcher = []
            else:
                pending_launcher = []

            entries.append(NormalizedEntry(record=record))

        return NormalizedTrace(entries=tuple(entries))

    def __open(self, *, target: str, steps: List[int]) -> NormalizedEntry:
        """
        Open the first real app: a grounded transition if launcher steps led here, else warm start.
        """

        if steps:
            return self.__transition(target=target, steps=steps)

        return NormalizedEntry(
            launch=LaunchMarker(package=target, provenance=LaunchProvenance.SYNTHETIC_WARM_START)
        )

    def __transition(self, *, target: str, steps: List[int]) -> NormalizedEntry:
        """
        Build a launcher-transition launch grounded by the collapsed launcher step numbers.
        """

        return NormalizedEntry(
            launch=LaunchMarker(
                package=target,
                provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                source_steps=tuple(steps),
            )
        )

    def __package(self, *, activity: Optional[str]) -> Optional[str]:
        """
        Extract the package base from an 'package/Activity' string, or None when absent.
        """

        if not activity:
            return None

        return activity.split("/", 1)[0].strip() or None
