from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, List

from fathom.core.prompts.templates import (
    VALIDATION_SUBJECT_EXTRACTION_SYSTEM,
    VALIDATION_SUBJECT_EXTRACTION_USER,
)
from fathom.core.services.normalizer import Normalizer
from fathom.utils.parsing import strip_code_fences

if TYPE_CHECKING:
    from fathom.interfaces.llm import LLMPort

logger = getLogger(__name__)


@dataclass
class ValidationSubjectsResult:
    """Result of validation subject extraction with provenance tracking."""

    subjects: list[str] = field(default_factory=list)
    source: str = "regex"  # "llm" or "regex"
    error: str | None = None


def extract_validation_subjects_regex(intent: str) -> list[str]:
    if not intent:
        return []

    matches = re.finditer(
        pattern=r"\b(?:validate|verify|check|confirm)(?:\s+that)?\s+(.+?)(?=(?:,|\bthen\b|\band\s+(?:validate|verify|check|confirm)\b|$))",
        string=intent,
        flags=re.IGNORECASE,
    )
    subjects: list[str] = []
    for match in matches:
        subject = Normalizer.clean(text=match.group(1).strip(" .,:;"))
        if subject:
            subjects.append(subject)

    return subjects


async def extract_validation_subjects_with_llm_tracked(
    *, llm: "LLMPort", intent: str
) -> ValidationSubjectsResult:
    """Extract validation subjects with source provenance tracking."""
    if not intent or not llm:
        return ValidationSubjectsResult(
            subjects=extract_validation_subjects_regex(intent=intent),
            source="regex",
            error="no intent or llm" if not intent else None,
        )

    try:
        response = await llm.generate(
            use_cache=False,
            prompt=[VALIDATION_SUBJECT_EXTRACTION_USER.format(intent=intent)],
            system_instruction=VALIDATION_SUBJECT_EXTRACTION_SYSTEM,
        )

        if not response or not response.content:
            logger.warning(
                "Gemini validation subject extraction returned empty response; using regex fallback."
            )
            return ValidationSubjectsResult(
                subjects=extract_validation_subjects_regex(intent=intent),
                source="regex",
                error="empty_llm_response",
            )

        try:
            content = strip_code_fences(response.content)

            subjects = json.loads(content)
            if not isinstance(subjects, list):
                logger.warning("Gemini returned non-list JSON; using regex fallback.")
                return ValidationSubjectsResult(
                    subjects=extract_validation_subjects_regex(intent=intent),
                    source="regex",
                    error="non_list_json",
                )

            normalized: List[str] = []
            for subject in subjects:
                if isinstance(subject, str):
                    cleaned = Normalizer.clean(text=subject.strip())
                    if cleaned:
                        normalized.append(cleaned)

            if normalized:
                logger.info(f"Gemini extracted {len(normalized)} validation subjects from intent.")
                return ValidationSubjectsResult(subjects=normalized, source="llm")
            else:
                logger.warning("Gemini extracted empty subjects; using regex fallback.")
                return ValidationSubjectsResult(
                    subjects=extract_validation_subjects_regex(intent=intent),
                    source="regex",
                    error="empty_llm_subjects",
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini JSON response ({e}); using regex fallback.")
            return ValidationSubjectsResult(
                subjects=extract_validation_subjects_regex(intent=intent),
                source="regex",
                error=f"json_decode: {e}",
            )

    except Exception as e:
        logger.warning(
            f"Gemini validation subject extraction failed ({e}); falling back to regex extraction."
        )
        return ValidationSubjectsResult(
            subjects=extract_validation_subjects_regex(intent=intent),
            source="regex",
            error=f"exception: {e}",
        )
