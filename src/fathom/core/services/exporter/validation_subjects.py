from __future__ import annotations

import json
import re
from logging import getLogger
from typing import TYPE_CHECKING, List

from fathom.core.services.normalizer import Normalizer

if TYPE_CHECKING:
    from fathom.interfaces.llm import LLMPort

logger = getLogger(__name__)


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


def extract_validation_subjects(intent: str) -> list[str]:
    return extract_validation_subjects_regex(intent=intent)


async def extract_validation_subjects_with_llm(*, llm: "LLMPort", intent: str) -> list[str]:
    if not intent or not llm:
        return extract_validation_subjects_regex(intent=intent)

    try:
        system_instruction = (
            "You are an expert at parsing user intents for mobile UI automation. "
            "Your task is to extract all validation requirements from a user's intent."
        )

        prompt = (
            f"Extract all validation requirements from this intent. "
            f"Return a JSON list of validation subjects (what to validate/confirm/check). "
            f"Each subject should be a complete, standalone assertion (e.g., 'the cart page is displayed', 'api validation succeeded'). "
            f"Handle numbered lists, conditionals, and complex sentences. Do not include keywords like 'Validate' or 'Check'. "
            f"Return ONLY valid JSON list of strings, no other text.\n\n"
            f"Intent: {intent}"
        )

        response = await llm.generate(
            use_cache=False,
            prompt=[prompt],
            system_instruction=system_instruction,
        )

        if not response or not response.content:
            logger.warning(
                "Gemini validation subject extraction returned empty response; using regex fallback."
            )
            return extract_validation_subjects_regex(intent=intent)

        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            subjects = json.loads(content)
            if not isinstance(subjects, list):
                logger.warning("Gemini returned non-list JSON; using regex fallback.")
                return extract_validation_subjects_regex(intent=intent)

            normalized: List[str] = []
            for subject in subjects:
                if isinstance(subject, str):
                    cleaned = Normalizer.clean(text=subject.strip())
                    if cleaned:
                        normalized.append(cleaned)

            if normalized:
                logger.info(f"Gemini extracted {len(normalized)} validation subjects from intent.")
                return normalized
            else:
                logger.warning("Gemini extracted empty subjects; using regex fallback.")
                return extract_validation_subjects_regex(intent=intent)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini JSON response ({e}); using regex fallback.")
            return extract_validation_subjects_regex(intent=intent)

    except Exception as e:
        logger.warning(
            f"Gemini validation subject extraction failed ({e}); falling back to regex extraction."
        )
        return extract_validation_subjects_regex(intent=intent)
