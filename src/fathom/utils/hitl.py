from __future__ import annotations

from typing import Callable, Optional, Tuple

from fathom.schemas.actions import Action
from fathom.services.ux import UXService
from fathom.utils.cli_input import input_with_lock


def prompt_human_review(
    action: Action,
    step_number: int,
    screen_description: Optional[str],
    current_intent: str,
    update_intent: Callable[[str], None],
) -> Tuple[Optional[Action], bool, bool]:
    """Prompt a human to approve, edit, or reject an action.

    Returns the approved or edited action. Returns None if rejected.
    """

    UXService().render_hitl_prompt(
        step_number=step_number,
        action=action.to_description(),
        rationale=action.rationale,
        current_intent=current_intent,
        screen_description=screen_description,
        decision_keys="a=approve, e=edit intent, r=exit session",
    )

    while True:
        choice = input_with_lock("Decision [a=approve, e=edit, r=exit]: ").strip().lower()
        if choice in ("", "a", "approve"):
            return action, False, False
        if choice in ("r", "exit"):
            return None, False, True
        if choice in ("e", "edit"):
            _prompt_edit_intent(current_intent=current_intent, update_intent=update_intent)
            return action, True, False

        print("Invalid choice. Enter a, e, or r.")


def _prompt_edit_intent(*, current_intent: str, update_intent: Callable[[str], None]) -> None:
    """Collect an updated intent from stdin."""

    new_intent = input_with_lock("New intent (blank to keep current): ").strip()
    if new_intent:
        update_intent(new_intent)
        return

    update_intent(current_intent)
