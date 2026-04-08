from __future__ import annotations

import re
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants import ActionType

# Matches the filler word "element" as a standalone token (not inside
# "elements" or "elementary"). The validate-action guard uses this to
# reject prose like "HealthTap homepage content, element visible" that
# leaks from the prompt's own vocabulary.
_FORBIDDEN_VALIDATION_SUBJECT_TOKEN = re.compile(r"\belement\b", re.IGNORECASE)

# Single source of truth for "this string is a placeholder, not a real
# target". Used by ExecuteAction normalization, the trace exporter, and
# the per-step Normalizer to reject prose like "element"/"button"/"icon".
GENERIC_TARGET_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "element",
        "ui element",
        "none",
        "label",
        "unknown",
        "a visible item",
        "button",
        "icon",
        "field",
        "text",
    }
)

# Prefixes the validate-action validator strips before deciding whether
# the resulting subject is empty. Public so adapter/parser code that
# constructs validate Actions can sanitize free-form input the same way.
VALIDATION_SUBJECT_BAD_PREFIXES: tuple[str, ...] = (
    "i am ",
    "i can ",
    "i will ",
    "i do ",
    "i have ",
    "i need ",
    "i want ",
    "validating ",
    "checking ",
    "confirming ",
    "the presence of ",
)


def clean_validation_subject(candidate: Optional[str], *, fallback: str) -> str:
    """Coerce free-form prose into a clean third-person noun phrase.

    Strips the first-person/narrative prefixes the Action validator
    forbids, keeps only the first sentence, and caps to ~8 words.
    Returns ``fallback`` when nothing usable remains.

    Rejection rules applied before returning:

    1. Empty / whitespace-only input → ``fallback``.
    2. After normalization, if the result lands on a member of
       :data:`GENERIC_TARGET_PLACEHOLDERS` (``"element"``, ``"button"``,
       ``"icon"``, ``"unknown"``, …) → ``fallback``. The
       ``Action._enforce_validation_subject`` validator enforces the
       same rule at construction time, but several call sites feed
       ``clean_validation_subject`` into dict-shaped payloads and
       non-Action consumers that bypass the model boundary, so the
       sanitizer has to catch the filler itself.
    3. If the normalized result still contains the standalone
       ``"element"`` token (e.g. ``"search box element"`` or
       ``"home screen, element visible"``) → ``fallback``. This is
       the same :data:`_FORBIDDEN_VALIDATION_SUBJECT_TOKEN` regex the
       Action validator uses, applied earlier in the pipeline.
    """

    text = (candidate or "").strip()
    if not text:
        return fallback

    previous = ""
    while text and text != previous:
        previous = text
        lower = text.lower()
        for prefix in VALIDATION_SUBJECT_BAD_PREFIXES:
            if lower.startswith(prefix):
                text = text[len(prefix) :].strip()
                break

    for terminator in (".", "!", "?", ";", "\n"):
        if terminator in text:
            text = text.split(terminator, 1)[0].strip()

    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])

    if not text:
        return fallback

    if text.lower() in GENERIC_TARGET_PLACEHOLDERS:
        return fallback

    if _FORBIDDEN_VALIDATION_SUBJECT_TOKEN.search(text):
        return fallback

    return text


def is_resolved_target(value: Any) -> bool:
    """True when *value* names a concrete UI subject (not a placeholder).

    Canonical check used by every target-resolution site in the
    codebase. A value is "resolved" when it is a non-empty string whose
    lowercased form is NOT in :data:`GENERIC_TARGET_PLACEHOLDERS`. The
    frozenset includes both "element" (the historic filler word the
    export pipeline keeps trying to eradicate) and "unknown" (the
    canonical fallback returned by :func:`resolve_action_target` when
    no candidate field carries real content), so this helper rejects
    both in one pass.
    """

    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in GENERIC_TARGET_PLACEHOLDERS


def _normalize_action_type(action_type: Any) -> str:
    """Coerce an ActionType enum or raw string to a lowercase string.

    Both ``Action`` (which uses the :class:`ActionType` enum) and
    ``ExecuteAction`` (which uses a raw string) feed into
    :func:`resolve_action_target`, so the router needs to tolerate
    both shapes without importing ``ActionType`` at the call site.
    """

    if action_type is None:
        return ""
    value = getattr(action_type, "value", action_type)
    return str(value).strip().lower()


def resolve_action_target(
    *,
    action_type: Any = None,
    target_name: Optional[str] = None,
    export_target: Optional[str] = None,
    natural_language_target: Optional[str] = None,
    validation_subject: Optional[str] = None,
    wait_subject: Optional[str] = None,
    scroll_target: Optional[str] = None,
    label_id: Optional[str] = None,
    fallback: str = "unknown",
) -> str:
    """Single source of truth for "what does this action point at?".

    Different action kinds store their authoritative subject in
    different fields, and before this helper existed the codebase had
    three independent routing implementations with subtly different
    fallback orders (``trace_payload._resolve_target``,
    ``prompts.trace.extract_action_fields``, and
    ``Action.to_description``). This function is the canonical
    resolver that all of them now delegate to.

    Routing rules:

    * ``validate`` → ``validation_subject``
    * ``wait``     → ``wait_subject``
    * ``scroll`` or any ``swipe_*`` → ``scroll_target``
    * everything else → skip straight to the general chain

    General chain (tried in order for every action kind after the
    kind-specific field above fails to resolve):

    1. ``target_name`` — the canonical field on ``ExecuteAction``.
    2. ``export_target`` — the exporter's preferred display name.
    3. ``natural_language_target`` — the legacy human-readable field
       still populated by the resolver and a few older code paths.
    4. ``f"label:{label_id}"`` — namespaced placeholder when the LLM
       only supplied a manifest label ID without a human-readable name.
    5. ``fallback`` — defaults to ``"unknown"``, which is itself in
       :data:`GENERIC_TARGET_PLACEHOLDERS` so downstream consumers
       that call :func:`is_resolved_target` on the result will
       correctly skip it rather than writing the filler into a
       script line or history entry.

    Every candidate is filtered through :func:`is_resolved_target`,
    so placeholder strings like ``"element"`` or ``"button"`` are
    treated as if the field were blank and the chain continues.
    """

    kind = _normalize_action_type(action_type)

    candidates: list[Optional[str]] = []
    if kind == "validate":
        candidates.append(validation_subject)
    elif kind == "wait":
        candidates.append(wait_subject)
    elif "swipe" in kind or kind == "scroll":
        candidates.append(scroll_target)

    candidates.extend(
        (
            target_name,
            export_target,
            natural_language_target,
        )
    )

    for candidate in candidates:
        if is_resolved_target(candidate):
            return str(candidate).strip()

    label = (label_id or "").strip()
    if label:
        return f"label:{label}"

    return fallback


class Bounds(BaseModel):
    """
    Bounds for UI elements.
    """

    x: int = Field(ge=0, le=5000, description="Top-left X coordinate")
    y: int = Field(ge=0, le=5000, description="Top-left Y coordinate")
    width: int = Field(ge=0, le=5000, description="Width of the element")
    height: int = Field(ge=0, le=5000, description="Height of the element")
    system: str = Field(
        default="normalized", description="Coordinate system used", alias="coord_system"
    )

    @property
    def is_normalized(self) -> bool:
        """
        Heuristic to check if coordinates are likely normalized (0-1000).
        """

        return self.x <= 1000 and self.y <= 1000 and self.width <= 1000 and self.height <= 1000

    @property
    def center_x(self) -> int:
        """
        Calculates the horizontal center.
        """

        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """
        Calculates the vertical center.
        """

        return self.y + self.height // 2

    def to_pixels(self, screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
        """
        Converts coordinates to absolute device pixels.
        Handles both normalized and already-pixel coordinates.
        Clamps results to valid screen bounds.
        """

        # If explicitly told it's pixels, don't normalize
        if self.system == "pixel":
            x, y, width, height = self.x, self.y, self.width, self.height
        elif self.is_normalized:
            x = int(self.x * screen_width / 1000)
            y = int(self.y * screen_height / 1000)
            width = int(self.width * screen_width / 1000)
            height = int(self.height * screen_height / 1000)
        else:
            # Fallback for large values that must be pixels
            x, y, width, height = self.x, self.y, self.width, self.height

        max_x = max(0, screen_width - 1)
        max_y = max(0, screen_height - 1)
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        width = max(1, min(width, max(1, screen_width - x)))
        height = max(1, min(height, max(1, screen_height - y)))

        return x, y, width, height


class Action(BaseModel):
    """
    Represents an atomic action to be performed on the mobile device.
    """

    action_type: ActionType = Field(description="The type of interaction to perform")

    rationale: str = Field(description="The reasoning behind choosing this action")
    target: str = Field(default="unknown", description="Grounding label ID or technical target")
    natural_language_target: Optional[str] = Field(
        default=None, description="Human-friendly name of the target element."
    )

    text: Optional[str] = Field(default=None, description="Text content for typing actions")
    bounds: Optional[Bounds] = Field(default=None, description="Bounding box for the interaction")
    label_id: Optional[str] = Field(default=None, description="Numeric label ID from XML grounding")

    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    wait_duration: Optional[float] = Field(default=None, description="Duration to wait in seconds")
    memory_updates: Optional[Dict[str, str]] = Field(
        default=None, description="Key-value pairs to store in persistent memory"
    )

    # Inline Validation
    is_valid: bool = Field(default=True, description="Self-validation of the action")

    # Conditional Execution
    condition: Optional[str] = Field(
        default=None,
        description="Condition required (e.g. 'Popup is visible', 'Section is collapsed', 'Error displayed')",
    )
    is_conditional: bool = Field(
        default=False,
        description="True when the action should execute only under a visible guard condition.",
    )
    conditional_type: Optional[Literal["blocker", "transient", "error", "optional"]] = Field(
        default=None,
        description="Optional conditional category: blocker, transient, error, or optional.",
    )
    overlay_detected: bool = Field(
        default=False,
        description="True when this action is specifically handling an overlay/popup blocker.",
    )

    # Script export classification (VLM-provided; optional; fallback is TargetClassifier)
    target_type: Optional[Literal["stable", "positional", "dynamic"]] = Field(
        default=None,
        description="How the target should be referenced in exported scripts: stable (fixed label), positional (ordinal in list), or dynamic (content that may change). Leave unset if unsure.",
    )
    script_target: Optional[str] = Field(
        default=None,
        description="When target_type is positional or dynamic, the exact phrase for script export (e.g. 'the first search result', 'the promotional banner'). Omit for stable.",
    )

    # Launch semantics (optional; used to disambiguate launcher icon taps from regular taps)
    is_app_launcher: bool = Field(
        default=False,
        description="Set to true when this tap action is specifically intended to launch or focus the target app. Helps the exporter replace launcher taps with OPEN_APP semantics.",
    )

    # Structured signal details for export (VLM-provided; authoritative)
    export_target: Optional[str] = Field(
        default=None,
        description=(
            "Canonical phrase for this action in exported test scripts. Must be specific "
            "and human-readable (e.g., 'Search box', 'the first search result', 'Add to cart button'). "
            "NEVER use generic placeholders like 'element', 'button', 'label'."
        ),
    )
    scroll_target: Optional[str] = Field(
        default=None,
        description="For scroll/swipe actions: the element or section being scrolled to find (e.g., 'Vitamins and supplements', 'Lab tests and packages'). Use the exact phrase from the UI when possible.",
    )
    wait_subject: Optional[str] = Field(
        default=None,
        description="For wait actions: what we're waiting for (e.g., 'app to load', 'search results to appear', 'Home page content'). Describe the expected state or element.",
    )
    validation_subject: Optional[str] = Field(
        default=None,
        description="For validate actions: what specifically is being validated (e.g., 'login status', 'banner visibility', 'item alignment'). Be specific about the validation target.",
    )
    target_is_generic: Optional[bool] = Field(
        default=None,
        description="Set to true when this action taps/selects a non-specific target (e.g., 'any item', 'random category', 'first result'). Signals that target should be generalized in export.",
    )
    target_element_type: Optional[
        Literal["button", "icon", "option", "link", "field", "text", "checkbox"]
    ] = Field(
        default=None,
        description="For tap/interact actions: the element type/role (button, icon, option, etc.). Helps refine target descriptions when product-specific elements are tapped.",
    )
    validation_pattern: Optional[Literal["blocker", "transient", "error", "generic"]] = Field(
        default=None,
        description="For validate actions: the pattern category - blocker (permission/popup/consent), transient (loading/spinner), error (network/validation error), or generic check.",
    )
    wait_pattern: Optional[Literal["ad", "splash", "load", "search", "generic"]] = Field(
        default=None,
        description="For wait actions: the wait category - ad (ad to finish), splash (app splash screen), load (content loading), search (search results), or generic.",
    )

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _enforce_validation_subject(self) -> "Action":
        """Validate actions MUST carry a short, third-person validation_subject.

        Without this guarantee, exporters are forced to fall back to
        free-form ``rationale`` prose (e.g. "I am validating the presence
        of the Categories tab...") which produces noisy script lines like
        "Validate I am validating the presence of...". Enforcing the
        subject at the core model boundary means every code path that
        constructs an Action — LLM tool boundary, replay, checkpoint
        restore, tests — is caught uniformly.
        """

        if self.action_type != ActionType.VALIDATE:
            return self

        subject = (self.validation_subject or "").strip()
        if not subject:
            raise ValueError(
                "validation_subject is required for action_type='validate'. "
                "Provide a short third-person noun phrase describing what is "
                "being checked (e.g., 'Instamart tab selected', 'cart is empty')."
            )

        lower = subject.lower()
        if any(lower.startswith(prefix) for prefix in VALIDATION_SUBJECT_BAD_PREFIXES):
            raise ValueError(
                f"validation_subject must not use first-person or narrative "
                f"prose: '{subject}'. Use a short third-person noun phrase "
                "(e.g., 'Instamart tab selected', 'cart is empty')."
            )

        if _FORBIDDEN_VALIDATION_SUBJECT_TOKEN.search(subject):
            raise ValueError(
                f"validation_subject must not contain the filler word "
                f"'element': '{subject}'. Name the actual thing being "
                "checked (e.g., 'Submit button enabled', "
                "'Cart total visible', 'Home tab selected')."
            )

        return self

    def to_description(self) -> str:
        """
        Generates a human-readable description of the action.
        """

        # Route every subject through the canonical resolver so this
        # helper, trace_payload._resolve_target, and
        # prompts.trace.extract_action_fields all share one chain.
        # The resolver handles per-kind routing, placeholder
        # skipping, and the label:{id} fallback.
        resolved = resolve_action_target(
            action_type=self.action_type,
            target_name=self.target,
            export_target=self.export_target,
            natural_language_target=self.natural_language_target,
            validation_subject=self.validation_subject,
            wait_subject=self.wait_subject,
            scroll_target=self.scroll_target,
            label_id=self.label_id,
        )

        if resolved.startswith("label:") and self.bounds:
            # We only know a manifest label ID and have pixel bounds
            # on hand — "Element at [x, y]" is historically more
            # readable than "label:7" in log output.
            name = f"Element at [{self.bounds.x}, {self.bounds.y}]"
        elif resolved.startswith("label:"):
            label_suffix = resolved.split(":", 1)[1]
            name = f"Element (Label {label_suffix})"
        elif resolved == "unknown":
            # Last-resort fallback — keep the historic "element"
            # literal here since it's purely a display string baked
            # into log/telemetry lines, never a field value.
            name = "element"
        else:
            name = resolved

        if self.action_type == ActionType.VALIDATE:
            return f"Validate {name}"

        if self.action_type == ActionType.TAP:
            return f"Tap on {name}"

        if self.action_type == ActionType.TYPE:
            text_val = self.text if self.text is not None else ""
            return f"Type '{text_val}' in {name}"

        if "swipe" in self.action_type.value:
            direction = (
                self.action_type.value.split("_")[-1]
                if "_" in self.action_type.value
                else "content"
            )
            return f"Swipe {direction} on {name}"

        if self.action_type == ActionType.SCROLL:
            return f"Scroll until you see {name}"

        if self.action_type == ActionType.LONG_PRESS:
            return f"Long press on {name}"

        if self.action_type == ActionType.BACK:
            return "Press back button"

        if self.action_type == ActionType.HOME:
            return "Press home button"

        if self.action_type == ActionType.WAIT:
            if self.wait_duration:
                return f"Wait for {self.wait_duration} seconds"
            return f"Wait for {name}"

        if self.action_type == ActionType.COMPLETE:
            return f"Validate {name} (Goal complete)"

        if self.action_type == ActionType.ASK_USER:
            msg = self.text or self.rationale or name
            return f"Ask user: {msg}"

        return f"{self.action_type.value.capitalize()} on {name}"


BoundingBox = Bounds
