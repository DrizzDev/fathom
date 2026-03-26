from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.exceptions import ScriptExportError
from fathom.core.prompts.export import ExportPromptBuilder
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.exporter.action_catalog import build_action_catalog_from_steps
from fathom.core.services.exporter.script_text import normalize_script_output
from fathom.core.services.exporter.structured_export import contains_goal_validation
from fathom.core.services.exporter.trace_payload import build_export_payload
from fathom.core.services.exporter.validation_subjects import (
    extract_validation_subjects_with_llm_tracked,
)
from fathom.interfaces.llm import LLMPort
from fathom.schemas.export import (
    ScriptExportStructuredPayload,
    ScriptExportStructuredPayloadShape,
)
from fathom.schemas.gemini_tools import EmitScriptArgs
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


class ScriptExporter:
    """
    Service for exporting execution history to natural language scripts.
    """

    def __init__(self, *, llm: Optional[LLMPort] = None, use_cache: bool = True) -> None:
        self.__llm = llm
        # use_cache is accepted for interface compatibility but not used:
        # export tool schemas contain dynamic enum values that change per call,
        # making Gemini context caching ineffective (zero reuse).
        _ = use_cache
        self.__prompt_builder: Optional[ExportPromptBuilder] = (
            PromptFactory.get_export_builder(model_name=llm.model_name) if llm else None
        )

    async def export_with_llm(
        self,
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        *,
        intent: str = "",
        goal_state: str = "",
        package_name: str = "",
    ) -> Optional[str]:
        if not self.__llm:
            return None

        payload = build_export_payload(step_results=step_results)
        if not payload:
            return None

        if not self.__prompt_builder:
            raise ScriptExportError("Script exporter prompt builder is not configured.")

        validation_result = await extract_validation_subjects_with_llm_tracked(
            llm=self.__llm, intent=(intent or goal_state)
        )
        validation_subjects = validation_result.subjects
        if validation_subjects:
            logger.info(
                f"Using Gemini-extracted validation subjects ({len(validation_subjects)} found) "
                "for LLM export baseline generation."
            )

        action_catalog, required_action_ids, required_open_app_id = build_action_catalog_from_steps(
            step_results=step_results,
            package_name=package_name,
            intent=(intent or goal_state),
        )
        action_catalog_lines = [
            f"{action_id}: {entry.description}" for action_id, entry in action_catalog.items()
        ]
        require_if_block = False

        system_instruction = self.__prompt_builder.build_system_instruction()
        prompt_text = self.__prompt_builder.build_user_prompt(
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            trace_payload=payload,
            action_catalog_lines=action_catalog_lines,
        )

        try:
            # use_cache=False: the tool schema now contains dynamic enum values
            # (action catalog IDs) that change per export, so caching the tools
            # would create a new Gemini cache entry every call with zero reuse.
            response = await self.__llm.generate(
                use_cache=False,
                prompt=[prompt_text],
                tools=ToolRegistry.get_export_definitions(action_ids=required_action_ids),
                system_instruction=system_instruction,
            )
        except Exception as exception:
            logger.warning(
                "Gemini script generation failed: %s [export_violation=llm_generation].", exception
            )
            return None

        structured_args: Dict[str, Any] = {}
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if getattr(tool_call, "name", "") != "emit_script":
                    continue
                arguments = getattr(tool_call, "args", {}) or {}
                structured_args = dict(arguments)
                break

        if not structured_args:
            logger.warning(
                "Gemini emit_script arguments missing [export_violation=missing_tool_args]."
            )
            return None

        try:
            raw_emit_args = EmitScriptArgs.model_validate(structured_args)
        except Exception as exception:
            logger.warning(
                "Gemini emit_script raw payload parsing failed: %s [export_violation=raw_parse].",
                exception,
            )
            return None

        try:
            shape = ScriptExportStructuredPayloadShape.model_validate(
                raw_emit_args.model_dump(exclude_unset=True)
            )
            # Convert CatalogEntry → str for the Pydantic export schema which
            # operates on rendered text (action_catalog is Dict[str, str] there).
            action_catalog_strings = {
                action_id: entry.description for action_id, entry in action_catalog.items()
            }
            structured_payload = ScriptExportStructuredPayload.enforce_policy(
                shape=shape,
                action_catalog=action_catalog_strings,
                required_action_ids=required_action_ids,
                required_open_app_id=required_open_app_id,
                require_if_block=require_if_block,
                expected_validation_count=len(validation_subjects),
            )
        except Exception as exception:
            logger.warning(
                "Gemini structured payload validation failed: %s "
                "[export_violation=structured_payload].",
                exception,
            )
            return None

        # Render to text. The structured payload is already fully validated by
        # enforce_policy() — action coverage, ordering, IF blocks, validations are
        # all checked there. to_script() is deterministic and always produces
        # well-formed output (balanced braces, correct ordering). No need to
        # re-parse the rendered text through ScriptExportPayload or is_valid_llm_script.
        candidate = normalize_script_output(script=structured_payload.to_script())
        if not contains_goal_validation(script=candidate):
            logger.warning(
                "Gemini script missing final goal validation "
                "[export_violation=missing_goal_validation]."
            )
            return None

        logger.info(
            "Gemini script export succeeded via structured payload [export_mode=llm_structured]."
        )
        return candidate
