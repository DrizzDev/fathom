from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.services.exporter.constants import (
    EXECUTABLE_ACTION_PREFIXES,
    SWIPE_ACTIONS,
    VALIDATE_PREFIX,
)
from fathom.core.services.exporter.step_record import (
    get_action_type,
    get_activity,
    is_launcher_activity,
    swipe_direction_label,
)
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


def _get_field(step: Union[StepResult, Dict[str, Any]], field: str, default: Any = None) -> Any:
    """Extract a field from a StepResult or dict, reading from the Action when available."""
    if isinstance(step, StepResult):
        return getattr(step.step.action, field, default)
    return step.get(field, default)


def build_action_catalog_from_steps(
    step_results: Sequence[Union[StepResult, Dict[str, Any]]],
    package_name: str,
    intent: str,
) -> tuple[Dict[str, str], list[str], Optional[str]]:
    lines: list[str] = []

    if package_name:
        lines.append(f"OPEN_APP {package_name}")

    n = len(step_results)
    i = 0
    while i < n:
        step = step_results[i]
        if is_launcher_activity(activity=get_activity(step)):
            i += 1
            continue
        action_type_val = get_action_type(step=step)

        # Use authoritative export_target from VLM; fall back to natural_language_target.
        export_target = _get_field(step, "export_target")
        if not export_target:
            export_target = _get_field(step, "natural_language_target") or "element"

        text = _get_field(step, "text")
        wait_duration = _get_field(step, "wait_duration")
        is_app_launcher_signal = _get_field(step, "is_app_launcher", False)
        wait_subject = _get_field(step, "wait_subject")
        scroll_target = _get_field(step, "scroll_target")

        # For wait actions, use authoritative wait_subject as the target.
        if action_type_val == "wait" and wait_subject:
            export_target = wait_subject

        if action_type_val in SWIPE_ACTIONS:
            swipe_direction = action_type_val
            j = i + 1
            while j < n and get_action_type(step=step_results[j]) == swipe_direction:
                j += 1
            i = j

            # Use authoritative scroll_target; fall back to next step's target.
            if scroll_target:
                visible_target = scroll_target
            elif i < n:
                next_export = _get_field(step_results[i], "export_target")
                visible_target = (
                    next_export
                    or _get_field(step_results[i], "natural_language_target")
                    or intent
                    or "the target"
                )
            else:
                visible_target = intent or "the target"

            label = swipe_direction_label(action_type=swipe_direction)
            lines.append(f"{label} until {visible_target} is visible")
            continue

        description = Normalizer.action(
            action_type=action_type_val,
            target=export_target,
            text=text,
            wait_duration=wait_duration,
        )

        lowered = description.lower()
        if lowered.startswith(VALIDATE_PREFIX):
            i += 1
            continue

        # Trust is_app_launcher exclusively — no heuristic fallback.
        is_first_step_derived_action = (
            bool(package_name) and len(lines) == 1 and lines[0].lower().startswith("open_app ")
        )
        if is_first_step_derived_action and action_type_val == "tap" and is_app_launcher_signal:
            logger.debug(
                "[EXPORTER] Collapsing launcher tap into OPEN_APP: target='%s' package=%s",
                export_target,
                package_name,
            )
            i += 1
            continue

        lines.append(description)
        i += 1

    executable_lines = [
        line.strip()
        for line in lines
        if line.strip() and line.strip().lower().startswith(EXECUTABLE_ACTION_PREFIXES)
    ]

    action_catalog: Dict[str, str] = {}
    required_action_ids: list[str] = []
    required_open_app_id: Optional[str] = None

    for index, line in enumerate(executable_lines, start=1):
        action_id = f"A{index}"
        action_catalog[action_id] = line
        required_action_ids.append(action_id)
        if required_open_app_id is None and line.lower().startswith("open_app "):
            required_open_app_id = action_id

    return action_catalog, required_action_ids, required_open_app_id
