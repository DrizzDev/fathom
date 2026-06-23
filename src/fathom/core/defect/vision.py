"""
Vision-based defect detection over a single screen's screenshot.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any, List, Optional

from fathom.constants.defect import (
    DETECT_DEFECTS_TOOL,
    DefectSeverity,
    DefectSignal,
    DefectSource,
)
from fathom.core.prompts.defect import DefectPromptBuilder
from fathom.core.prompts.tools import ToolRegistry
from fathom.interfaces.defect import ScreenDefectDetectorPort
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.actions import Bounds
from fathom.schemas.defect import Defect, DefectEvidence, ScreenSnapshot
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class VisionDefectDetector(ScreenDefectDetectorPort):
    """
    Detects user-visible UI and content defects from a screen's screenshot via an LLM.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        use_cache: bool = True,
        builder: Optional[DefectPromptBuilder] = None,
    ) -> None:
        self.__llm = llm
        self.__use_cache = use_cache
        self.__builder = builder or DefectPromptBuilder()

    async def inspect_screen(self, *, snapshot: ScreenSnapshot) -> List[Defect]:
        """
        Returns the defects the model reports on the screenshot, or none when no image.
        """

        if snapshot.screenshot is None:
            return []

        prompt: List[PromptPart] = [
            f"Inspect this {snapshot.activity or 'app'} screen for user-visible defects.",
            snapshot.screenshot,
        ]
        result = await self.__llm.generate(
            use_cache=self.__use_cache,
            prompt=prompt,
            tools=ToolRegistry.get_defect_definitions(),
            system_instruction=self.__builder.build_system_prompt(),
        )
        return self.__parse(result=result, snapshot=snapshot)

    def __parse(self, *, result: GenerateResult, snapshot: ScreenSnapshot) -> List[Defect]:
        """
        Maps a detect_defects tool call into anchored defects, skipping malformed entries.
        """

        defects: List[Defect] = []
        for call in result.tool_calls:
            if getattr(call, "name", "") != DETECT_DEFECTS_TOOL:
                continue
            findings = dict(getattr(call, "args", {}) or {}).get("defects", [])
            if not isinstance(findings, list):
                continue
            for finding in findings:
                defect = self.__to_defect(finding=finding, snapshot=snapshot)
                if defect is not None:
                    defects.append(defect)
        return defects

    @classmethod
    def __to_defect(cls, *, finding: Any, snapshot: ScreenSnapshot) -> Optional[Defect]:
        """
        Builds one defect from a finding object, or None when it is unusable.
        """

        if not isinstance(finding, dict):
            return None

        signal = cls.__coerce_signal(finding.get("signal"))
        summary = str(finding.get("summary") or "").strip()
        if signal is None or not summary:
            return None

        return Defect.from_signal(
            signal=signal,
            source=DefectSource.POST_RUN,
            summary=summary,
            severity=cls.__coerce_severity(finding.get("severity")),
            evidence=DefectEvidence(
                screen=snapshot.screen,
                activity=snapshot.activity,
                bounds=cls.__coerce_bounds(finding.get("bounds")),
                excerpt=summary,
            ),
        )

    @staticmethod
    def __coerce_signal(value: Any) -> Optional[DefectSignal]:
        """
        Coerces a raw signal string into the enum, or None when invalid.
        """

        try:
            return DefectSignal(value)
        except ValueError:
            return None

    @staticmethod
    def __coerce_severity(value: Any) -> Optional[DefectSeverity]:
        """
        Coerces a raw severity string into the enum, or None to use the signal default.
        """

        try:
            return DefectSeverity(value)
        except ValueError:
            return None

    @staticmethod
    def __coerce_bounds(value: Any) -> Optional[Bounds]:
        """
        Builds normalized bounds from a raw box, or None when absent or malformed.
        """

        if not isinstance(value, dict):
            return None
        try:
            return Bounds(
                x=int(value.get("x", 0)),
                y=int(value.get("y", 0)),
                width=int(value.get("width", 0)),
                height=int(value.get("height", 0)),
            )
        except (TypeError, ValueError):
            return None
