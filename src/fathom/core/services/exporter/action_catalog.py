from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.services.exporter.constants import GENERIC_TARGETS, SWIPE_ACTIONS
from fathom.core.services.exporter.script_text import sanitize_script_targets
from fathom.core.services.exporter.step_inference import infer_scroll_target, infer_wait_subject
from fathom.core.services.exporter.step_record import (
    get_action_type,
    get_activity,
    is_launcher_activity,
    swipe_direction_label,
)
from fathom.core.services.exporter.step_targets import (
    extract_goal_label,
    is_likely_launch_tap,
    resolve_target,
)
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


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
        target = resolve_target(step=step)

        if isinstance(step, StepResult):
            text = step.step.action.text
            rationale = step.step.action.rationale
            wait_duration = step.step.action.wait_duration
            is_app_launcher_signal = step.step.action.is_app_launcher
            wait_subject = step.step.action.wait_subject
            wait_pattern = step.step.action.wait_pattern
            scroll_target = step.step.action.scroll_target
        else:
            text = step.get("text")
            rationale = str(object=step.get("rationale", ""))
            wait_duration = step.get("wait_duration")
            is_app_launcher_signal = step.get("is_app_launcher", False)
            wait_subject = step.get("wait_subject")
            wait_pattern = step.get("wait_pattern")
            scroll_target = step.get("scroll_target")

        if action_type_val == "wait" and target.lower() in GENERIC_TARGETS:
            if wait_subject:
                target = wait_subject
            elif wait_pattern:
                pattern_map = {
                    "ad": "ad to finish",
                    "splash": "app to finish loading",
                    "load": "app to finish loading",
                    "search": "search results to appear",
                }
                target = pattern_map.get(wait_pattern, "screen to load")
            else:
                target = infer_wait_subject(rationale=rationale, wait_subject=None)

        if action_type_val in SWIPE_ACTIONS:
            swipe_direction = action_type_val
            swipe_start = i
            j = i + 1
            while j < n and get_action_type(step=step_results[j]) == swipe_direction:
                j += 1
            i = j

            if i < n:
                next_target = resolve_target(step=step_results[i])
            else:
                next_target = (
                    scroll_target
                    or infer_scroll_target(steps=step_results, start=swipe_start, end=i)
                    or extract_goal_label(goal_state=intent)
                    or intent
                    or "the target"
                )

            label = swipe_direction_label(action_type=swipe_direction)
            lines.append(f"{label} until {next_target} is visible")
            continue

        description = Normalizer.action(
            action_type=action_type_val, target=target, text=text, wait_duration=wait_duration
        )

        lowered = description.lower()
        if lowered.startswith("validate "):
            i += 1
            continue

        is_first_step_derived_action = (
            bool(package_name) and len(lines) == 1 and lines[0].lower().startswith("open_app ")
        )
        is_launch_tap = (
            is_first_step_derived_action
            and action_type_val == "tap"
            and (
                is_app_launcher_signal
                or is_likely_launch_tap(
                    target=target,
                    description=description,
                )
            )
        )
        if is_launch_tap:
            logger.debug(
                f"[EXPORTER] Collapsing launcher tap into OPEN_APP: target='{target}' "
                f"description='{description}' package={package_name} "
                f"launcher_signal={is_app_launcher_signal}"
            )
            i += 1
            continue

        lines.append(description)
        i += 1

    executable_prefixes = (
        "open_app ",
        "tap ",
        "type ",
        "scroll ",
        "swipe ",
        "wait ",
        "press ",
        "long press ",
    )
    sanitized_script = sanitize_script_targets(
        script="\n".join(lines) + ("\n" if lines else ""),
        intent=intent,
    )
    executable_lines = [
        line.strip()
        for line in sanitized_script.splitlines()
        if line.strip() and line.strip().lower().startswith(executable_prefixes)
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
