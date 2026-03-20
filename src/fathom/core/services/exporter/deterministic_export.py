from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Union

from fathom.core.services.exporter.constants import GENERIC_TARGETS, SWIPE_ACTIONS
from fathom.core.services.exporter.script_text import sanitize_script_targets
from fathom.core.services.exporter.step_inference import (
    get_condition,
    infer_scroll_target,
    infer_validation_condition,
    infer_wait_subject,
    is_blocker_popup_condition,
    is_system_validation,
)
from fathom.core.services.exporter.step_record import (
    default_condition_for_type,
    find_app_launch_boundary,
    get_action_type,
    get_activity,
    get_conditional_type,
    get_event_type,
    get_raw_condition,
    infer_open_app_package,
    is_explicit_conditional,
    is_generic_wait_condition,
    is_launcher_activity,
    is_overlay_detected,
    swipe_direction_label,
)
from fathom.core.services.exporter.step_targets import (
    extract_goal_label,
    is_intent_target,
    is_likely_launch_tap,
    resolve_target,
)
from fathom.core.services.exporter.validation_subjects import extract_validation_subjects
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.steps import StepResult


def export_steps_to_script(
    step_results: Sequence[Union[StepResult, Dict[str, Any]]],
    *,
    intent: str = "",
    goal_state: str = "",
    package_name: str = "",
    include_final_validation: bool = True,
    validation_subjects_override: Optional[Sequence[str]] = None,
) -> str:
    lines: list[str] = []

    filtered_results = []
    for step in step_results:
        if get_condition(step=step) == "recovery":
            continue
        filtered_results.append(step)

    step_results = filtered_results

    n = len(step_results)
    i = 0
    launch_boundary = 0
    swipe_just_processed = False
    if validation_subjects_override is not None:
        validation_subjects = []
        for subject in validation_subjects_override:
            cleaned_subject = Normalizer.clean(text=str(subject).strip(" .,:;"))
            if cleaned_subject:
                validation_subjects.append(cleaned_subject)
    else:
        validation_subjects = extract_validation_subjects(intent=(intent or goal_state))
    validation_subject_index = 0
    reserved_final_subjects = 1 if include_final_validation and validation_subjects else 0
    emitted_validation_lines: Set[str] = set()

    if package_name:
        launch_boundary = find_app_launch_boundary(steps=step_results, package_name=package_name)
        if launch_boundary > 0:
            lines.append(f"OPEN_APP {package_name}")
            i = launch_boundary
        else:
            inferred_package = infer_open_app_package(
                steps=step_results, default_package=package_name
            )
            if inferred_package:
                lines.append(f"OPEN_APP {inferred_package}")
                i = 0

    while i < n:
        step = step_results[i]
        if is_launcher_activity(activity=get_activity(step)):
            i += 1
            continue
        action_type_val = get_action_type(step=step)
        event_type = get_event_type(step=step)
        raw_condition = get_raw_condition(step=step)
        condition = get_condition(step=step)
        explicit_conditional = is_explicit_conditional(step=step)
        conditional_type = get_conditional_type(step=step)
        if explicit_conditional:
            default_condition = default_condition_for_type(conditional_type=conditional_type)
            if default_condition and (
                not raw_condition or not condition or is_generic_wait_condition(condition=condition)
            ):
                condition = default_condition
        if is_overlay_detected(step=step) and not is_blocker_popup_condition(condition=condition):
            condition = "Overlay is visible"
        target = resolve_target(step=step)

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
                    infer_scroll_target(steps=step_results, start=swipe_start, end=i)
                    or extract_goal_label(goal_state=(intent or goal_state))
                    or intent
                    or goal_state
                    or "the target"
                )

            label = swipe_direction_label(action_type=swipe_direction)
            lines.append(f"{label} until {next_target} is visible")
            swipe_just_processed = True
            continue

        deferred_screen_validation: Optional[List[str]] = None
        if (
            i > 0
            and i > launch_boundary
            and action_type_val != "wait"
            and not swipe_just_processed
            and event_type != "validation"
            and not condition
        ):
            prev = step_results[i - 1]
            prev_action_type = get_action_type(step=prev)
            prev_condition = get_condition(step=prev)
            prev_changed = (
                prev.screen_changed
                if isinstance(prev, StepResult)
                else prev.get("screen_changed", False)
            )
            if (
                prev_changed
                and prev_action_type not in ("wait", *SWIPE_ACTIONS)
                and target.lower() not in GENERIC_TARGETS
            ):
                available_for_intermediate = max(
                    0, len(validation_subjects) - reserved_final_subjects
                )
                if validation_subject_index < available_for_intermediate:
                    requested_subject = validation_subjects[validation_subject_index]
                    validation_subject_index += 1
                    val_line = Normalizer.validation(target=requested_subject, explicit=True)
                else:
                    val_line = f"Validate {target} is visible"
                if prev_condition:
                    deferred_screen_validation = [
                        f"IF {prev_condition}",
                        "{",
                        f"    {val_line}",
                        "}",
                    ]
                else:
                    deferred_screen_validation = [val_line]

        swipe_just_processed = False

        if isinstance(step, StepResult):
            action = step.step.action
            text = action.text
            rationale = action.rationale
            wait_duration = action.wait_duration
            is_app_launcher_signal = action.is_app_launcher
            wait_subject = action.wait_subject
            wait_pattern = action.wait_pattern
        else:
            text = step.get("text")
            rationale = str(object=step.get("rationale", ""))
            wait_duration = step.get("wait_duration")
            is_app_launcher_signal = bool(step.get("is_app_launcher", False))
            wait_subject = step.get("wait_subject")
            wait_pattern = step.get("wait_pattern")

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

        description = Normalizer.action(
            action_type=action_type_val, target=target, text=text, wait_duration=wait_duration
        )

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
            i += 1
            continue

        if event_type == "validation":
            effective_target = target
            is_sys_validation = is_system_validation(
                target=target, rationale=rationale, condition=condition
            )
            should_use_intent_subject = (
                bool(validation_subjects)
                and not is_sys_validation
                and validation_subject_index
                < max(0, len(validation_subjects) - reserved_final_subjects)
            )
            if should_use_intent_subject:
                effective_target = validation_subjects[
                    min(validation_subject_index, len(validation_subjects) - 1)
                ]
                validation_subject_index += 1
            validation_condition = infer_validation_condition(
                condition=condition,
                action_type=action_type_val,
                target=effective_target,
                rationale=rationale,
            )
            if not validation_condition and i > 0 and not should_use_intent_subject:
                prev = step_results[i - 1]
                prev_condition = get_condition(step=prev)
                prev_action_type = get_action_type(step=prev)
                if prev_condition and prev_action_type == "wait":
                    validation_condition = prev_condition

            validation_line = Normalizer.validation(
                target=effective_target,
                explicit=should_use_intent_subject,
            )

            if validation_condition and i + 1 < n:
                next_step = step_results[i + 1]
                next_condition = get_condition(step=next_step)
                next_target = resolve_target(step=next_step)
                next_event_type = get_event_type(step=next_step)
                if (
                    next_event_type != "validation"
                    and next_condition == validation_condition
                    and next_target.strip().lower() == target.strip().lower()
                ):
                    i += 1
                    continue
            if validation_line in emitted_validation_lines:
                i += 1
                continue
            if validation_condition:
                merged_into_previous_wait_block = False
                if i > 0:
                    prev = step_results[i - 1]
                    prev_action_type = get_action_type(step=prev)
                    prev_condition = get_condition(step=prev)
                    if (
                        prev_action_type == "wait"
                        and prev_condition == validation_condition
                        and len(lines) >= 4
                        and lines[-4] == f"IF {validation_condition}"
                        and lines[-3] == "{"
                        and lines[-1] == "}"
                    ):
                        lines.pop()
                        lines.append(f"    {validation_line}")
                        lines.append("}")
                        merged_into_previous_wait_block = True
                if merged_into_previous_wait_block:
                    emitted_validation_lines.add(validation_line)
                    i += 1
                    continue
                lines.append(f"IF {validation_condition}")
                lines.append("{")
                lines.append(f"    {validation_line}")
                lines.append("}")
            else:
                lines.append(validation_line)
            emitted_validation_lines.add(validation_line)
            i += 1
            continue

        if condition and action_type_val != "wait":
            lines.append(f"IF {condition}")
            lines.append("{")
            prev_is_same_target_validation = False
            if i > 0:
                prev = step_results[i - 1]
                prev_event_type = get_event_type(step=prev)
                prev_target = resolve_target(step=prev)
                prev_is_same_target_validation = (
                    prev_event_type == "validation"
                    and prev_target.strip().lower() == target.strip().lower()
                )
            if (
                target.lower() not in GENERIC_TARGETS
                and action_type_val != "wait"
                and not prev_is_same_target_validation
            ):
                lines.append(f"    Validate {target} is visible")
            lines.append(f"    {description}")
            lines.append("}")
        else:
            lines.append(description)

        if deferred_screen_validation:
            suf = deferred_screen_validation
            if len(lines) < len(suf) or lines[-len(suf) :] != suf:
                lines.extend(suf)
        i += 1

    if include_final_validation and step_results:
        last_action_type = get_action_type(step=step_results[-1])

        if last_action_type not in ("complete", "verify_goal_completion"):
            if validation_subject_index < len(validation_subjects):
                final_subject = validation_subjects[validation_subject_index]
                final_validation_line = f"Validate that {final_subject}"
                if final_validation_line not in emitted_validation_lines and (
                    not lines or lines[-1].strip() != final_validation_line
                ):
                    lines.append(final_validation_line)
                    emitted_validation_lines.add(final_validation_line)
                script = "\n".join(lines) + "\n"
                return sanitize_script_targets(script=script, intent=(intent or goal_state))

            goal_label = extract_goal_label(goal_state=(intent or goal_state))
            last_target = resolve_target(step=step_results[-1])
            last_target_usable = last_target and last_target.lower() not in GENERIC_TARGETS

            if last_target_usable and is_intent_target(
                target=last_target, intent=(intent or goal_state)
            ):
                val_line = f"Validate {last_target} is visible"
                if not lines or lines[-1].strip() != val_line:
                    lines.append(val_line)
            elif goal_label:
                val_line = f"Validate {goal_label} is visible"
                if not lines or lines[-1].strip() != val_line:
                    lines.append(val_line)
            elif last_target_usable:
                val_line = f"Validate {last_target} is visible"
                if not lines or lines[-1].strip() != val_line:
                    lines.append(val_line)
            else:
                val_line = "Validate Goal State is visible"
                if not lines or lines[-1].strip() != val_line:
                    lines.append(val_line)

    script = "\n".join(lines) + "\n"
    return sanitize_script_targets(script=script, intent=(intent or goal_state))
