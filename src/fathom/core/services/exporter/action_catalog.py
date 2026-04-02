from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.services.exporter.constants import (
    EXECUTABLE_ACTION_PREFIXES,
    SWIPE_ACTIONS,
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


# ---------------------------------------------------------------------------
# OPEN_APP resolution — isolated so deep links, intents, multi-app flows
# can be extended here without touching the action catalog loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogEntry:
    """
    A single entry in the action catalog, preserving both the rendered
    description and the structured action kind from the source StepResult.

    This avoids re-parsing action types from rendered text downstream.
    """

    description: str  # Rendered action line (e.g. "TAP on 'Add to cart' button")
    action_kind: str  # Structured action type (e.g. "tap", "type", "scroll", "open_app")

    def __str__(self) -> str:
        return self.description


@dataclass(frozen=True)
class AppLaunchDescriptor:
    """
    Describes how to launch the target app.

    Extensible for future launch modes (deep links, intents, etc.).
    """

    command: str  # e.g. "OPEN_APP com.example.app"
    package: str  # e.g. "com.example.app"
    skip_steps: int = 0  # Number of leading steps consumed by the launch (launcher taps)


def resolve_app_launch(
    step_results: Sequence[Union[StepResult, Dict[str, Any]]],
    package_name: str,
) -> Optional[AppLaunchDescriptor]:
    """
    Determine the app launch command and how many leading steps it consumes.

    Currently handles:
    - Package-based OPEN_APP with launcher tap collapse.

    Future extensions:
    - Deep link / URI intent launches
    - Multi-app orchestration (launch app A then switch to app B)
    - App-not-installed detection
    """

    if not package_name:
        return None

    # Count leading launcher steps that should be collapsed into the OPEN_APP.
    skip = 0
    for step in step_results:
        if not is_launcher_activity(activity=get_activity(step)):
            # Check if the first non-launcher step is a tap on the app icon
            action_type = get_action_type(step=step)
            is_app_launcher_signal = _get_field(step, "is_app_launcher", False)
            if action_type == "tap" and (
                is_app_launcher_signal or is_launcher_activity(activity=get_activity(step))
            ):
                skip += 1
            break
        skip += 1

    return AppLaunchDescriptor(
        command=f"OPEN_APP {package_name}",
        package=package_name,
        skip_steps=skip,
    )


# ---------------------------------------------------------------------------
# Action catalog builder — converts step trace into ordered action lines.
# ---------------------------------------------------------------------------


def build_action_catalog_from_steps(
    step_results: Sequence[Union[StepResult, Dict[str, Any]]],
    package_name: str,
    intent: str,
) -> tuple[Dict[str, CatalogEntry], list[str], Optional[str]]:
    """
    Build an ordered action catalog from execution step results.

    Returns:
        (action_catalog, required_action_ids, required_open_app_id)

    Each catalog entry preserves the structured action kind alongside the
    rendered description, avoiding the need to re-parse action types from text.
    """

    entries: list[CatalogEntry] = []

    # Phase 1: Resolve app launch (isolated from the action loop).
    app_launch = resolve_app_launch(step_results=step_results, package_name=package_name)
    steps_to_skip = 0
    if app_launch:
        entries.append(CatalogEntry(description=app_launch.command, action_kind="open_app"))
        steps_to_skip = app_launch.skip_steps

    # Phase 2: Build action lines from remaining steps.
    n = len(step_results)
    i = steps_to_skip
    while i < n:
        step = step_results[i]
        if is_launcher_activity(activity=get_activity(step)):
            i += 1
            continue
        action_type_val = get_action_type(step=step)

        # For positional/dynamic targets, prefer script_target (e.g. "the 1st product")
        # over export_target (e.g. "R for Rabbit Pant Diaper") to keep scripts reusable.
        target_type = _get_field(step, "target_type")
        script_target = _get_field(step, "script_target")
        export_target = _get_field(step, "export_target")

        if target_type in ("positional", "dynamic") and script_target:
            export_target = script_target
        elif not export_target:
            export_target = _get_field(step, "natural_language_target") or "element"

        text = _get_field(step, "text")
        wait_duration = _get_field(step, "wait_duration")
        wait_subject = _get_field(step, "wait_subject")
        scroll_target = _get_field(step, "scroll_target")
        validation_subject = _get_field(step, "validation_subject")

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
            entries.append(
                CatalogEntry(
                    description=f"{label} until {visible_target} is visible",
                    action_kind="scroll",
                )
            )
            continue

        # For validate actions, ensure we have a meaningful subject.
        if action_type_val == "validate" and not validation_subject:
            validation_subject = (
                _get_field(step, "condition")
                or _get_field(step, "validation_reason")
                or _get_field(step, "rationale")
                or export_target
            )

        description = Normalizer.action(
            action_type=action_type_val,
            target=export_target,
            text=text,
            wait_duration=wait_duration,
            validation_subject=validation_subject,
        )

        # Skip "complete" actions — they are goal signals, not executable steps.
        if action_type_val == "complete":
            i += 1
            continue

        entries.append(CatalogEntry(description=description, action_kind=action_type_val))
        i += 1

    # Phase 3: Build the indexed catalog.
    executable_entries = [
        entry
        for entry in entries
        if entry.description.strip()
        and entry.description.strip().lower().startswith(EXECUTABLE_ACTION_PREFIXES)
    ]

    action_catalog: Dict[str, CatalogEntry] = {}
    required_action_ids: list[str] = []
    required_open_app_id: Optional[str] = None

    for index, entry in enumerate(executable_entries, start=1):
        action_id = f"A{index}"
        action_catalog[action_id] = entry
        # Validate actions are available in the catalog but not required —
        # the LLM may cover them via action_validations or final_validation.
        if entry.action_kind != "validate":
            required_action_ids.append(action_id)
        if required_open_app_id is None and entry.action_kind == "open_app":
            required_open_app_id = action_id

    return action_catalog, required_action_ids, required_open_app_id
