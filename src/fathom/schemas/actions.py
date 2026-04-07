from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants import ActionType

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

    return text or fallback


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
    target: str = Field(default="element", description="Grounding label ID or technical target")
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
    validation_reason: Optional[str] = Field(
        default=None, description="Reason if action is invalid"
    )

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

        return self

    def to_description(self) -> str:
        """
        Generates a human-readable description of the action.
        """

        # Resolve best target description.
        # For validate actions, prefer validation_subject over generic targets.
        if self.action_type == ActionType.VALIDATE and self.validation_subject:
            return f"Validate {self.validation_subject}"

        name = self.natural_language_target

        lowered = (name or "").strip().lower()
        if not name or lowered in ("element", "ui element", "none", "label", "unknown"):
            # Fallback to label ID or bounds if natural language target is generic/missing
            if self.label_id:
                name = f"Element (Label {self.label_id})"

            elif self.bounds:
                name = f"Element at [{self.bounds.x}, {self.bounds.y}]"

            else:
                name = self.target or "element"

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
