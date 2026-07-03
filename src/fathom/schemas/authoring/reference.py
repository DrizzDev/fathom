from __future__ import annotations

from typing import Tuple

from pydantic import Field

from fathom.constants.authoring import AuthoringExampleKind
from fathom.constants.dialect import DialectName
from fathom.schemas.base import SealedModel


class CommandExample(SealedModel):
    """
    Example that teaches command usage without changing runtime behavior.
    """

    reason: str = Field(min_length=1, description="Why this example is preferred or avoided.")
    command: str = Field(min_length=1, description="Rendered command pattern for the situation.")
    situation: str = Field(min_length=1, description="Evidence situation the example represents.")
    kind: AuthoringExampleKind = Field(description="Whether the example is preferred or avoided.")


class CommandDoc(SealedModel):
    """
    Command semantics and syntax exposed to authoring prompts.
    """

    name: str = Field(min_length=1, description="Command or node name.")
    purpose: str = Field(min_length=1, description="What the command means.")
    syntax: str = Field(min_length=1, description="Canonical rendered syntax.")
    example: str = Field(min_length=1, description="One valid command example.")

    rules: Tuple[str, ...] = Field(
        default_factory=tuple, description="Command-specific authoring constraints."
    )
    examples: Tuple[CommandExample, ...] = Field(
        default_factory=tuple, description="Few-shot examples for command usage."
    )


class DialectGuide(SealedModel):
    """
    Dialect-level authoring guidance shared by all commands.
    """

    principles: Tuple[str, ...] = Field(description="Core replayability principles.")
    selection: Tuple[str, ...] = Field(description="How to choose stable or dynamic targets.")

    composition: Tuple[str, ...] = Field(description="How to merge or separate commands.")
    completion: Tuple[str, ...] = Field(description="How to decide complete versus partial Flow.")


class AuthoringDialectReference(SealedModel):
    """
    Script dialect reference supplied to the authoring agent.
    """

    name: DialectName = Field(description="Target script dialect.")
    guide: DialectGuide = Field(description="Dialect-level authoring guide.")
    commands: Tuple[CommandDoc, ...] = Field(
        min_length=1, description="Commands supported by the target dialect."
    )


DRIZZ_GUIDE = DialectGuide(
    principles=(
        "Author a replayable script, not a literal transcript of every attempted step.",
        "Every command must be grounded in supplied evidence and valid for the target dialect.",
        "Prefer clear, complete command phrases that a future replay can execute without hidden context.",
        "Use recorded values for value-bearing commands; never invent values or assertions.",
    ),
    selection=(
        "Use an exact target when evidence identifies a stable semantic UI element.",
        "Use a relative or dynamic target when evidence shows selection by order, query, filter, condition, or runtime context.",
        "Combine stable and dynamic context when both are needed to make the target replayable.",
        "Avoid raw trace labels when they are generic, incomplete, or only meaningful to the recorder.",
    ),
    composition=(
        "Merge repeated attempts that serve one user-level purpose into one replayable command when the evidence supports it.",
        "Keep separate commands when evidence shows separate user-level purposes or different command semantics.",
        "Represent conditional behavior only when a condition and its branch were recorded.",
        "Do not turn internal memory updates, diagnostics, or observations into script commands.",
    ),
    completion=(
        "Return a complete Flow only when evidence proves the user-level goal was completed.",
        "Return a partial Flow when the run stopped, evidence is insufficient, or completion cannot be asserted faithfully.",
        "A validation command must assert a meaningful visible or data state, not merely repeat a generic control label.",
    ),
)


DRIZZ_COMMANDS: Tuple[CommandDoc, ...] = (
    CommandDoc(
        name="open_app",
        syntax="OPEN_APP: <package>",
        example="OPEN_APP: com.android.chrome",
        purpose="Launch the target application.",
        rules=("Use the target package recorded by launch evidence.",),
    ),
    CommandDoc(
        name="tap",
        syntax="Tap on <target>",
        example="Tap on Login button",
        purpose="Tap a replayable UI target.",
        rules=(
            "Target must identify what replay should tap.",
            "Prefer semantic labels over recorder-only labels when evidence supports the semantic label.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                situation="A search result was chosen by query and position.",
                command="Tap on first search result for the recorded search query",
                reason="The target combines relative selection with the query that made it meaningful.",
            ),
            CommandExample(
                command="Tap on first card",
                kind=AuthoringExampleKind.AVOID,
                situation="A recorded target only names an ordinal card.",
                reason="The command is grounded but too ambiguous for replay without selection context.",
            ),
        ),
    ),
    CommandDoc(
        name="type",
        syntax='Type "<value>" into <field>',
        example='Type "John" into name input field',
        purpose="Enter a value into a focused field.",
        rules=("Use the recorded typed value and the field it was entered into.",),
    ),
    CommandDoc(
        name="scroll",
        example="Scroll down",
        syntax="Scroll <direction>",
        purpose="Move through content in a direction.",
        rules=(
            "Use plain scroll when the evidence records movement but no concrete stop target.",
            "Direction is page-motion direction, not finger-motion direction.",
        ),
        examples=(
            CommandExample(
                command="Scroll down",
                kind=AuthoringExampleKind.PREFERRED,
                reason="Plain movement is faithful when no concrete stop target is proven.",
                situation="The run performed several scroll gestures while searching within one list.",
            ),
        ),
    ),
    CommandDoc(
        name="scroll_until",
        syntax="Scroll <direction> until <target>",
        example='Scroll down until "Ratings and Reviews section"',
        purpose="Move through content until a concrete target is found.",
        rules=(
            "Use only when evidence records a concrete target that replay should stop at.",
            "The target must be complete enough to tell replay what should become visible.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command='Scroll down until "Reviews section"',
                reason="The stop target is concrete and visible.",
                situation="Scrolling stopped when a concrete section became visible.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command='Scroll down until "result list"',
                situation="Several scroll attempts mention broad list context.",
                reason="The stop target is too broad to be a replayable stopping condition.",
            ),
        ),
    ),
    CommandDoc(
        name="wait",
        purpose="Wait for time or a state.",
        example="Wait until search results are visible",
        syntax="Wait for <seconds> seconds or Wait until <subject>",
        rules=("Prefer state-based waits over fixed time when evidence records a state.",),
    ),
    CommandDoc(
        name="store",
        syntax="Store <value> as <name>",
        example="Store 123456 as OTP code",
        purpose="Store a captured runtime value under a variable name.",
        rules=(
            "Use the captured value, not the capture subject.",
            "Emit Store only for successful captured values recorded in evidence.",
        ),
        examples=(
            CommandExample(
                command="Store Rs. 76 as item.price",
                kind=AuthoringExampleKind.PREFERRED,
                reason="The command stores the actual runtime value.",
                situation="Evidence captured a runtime value for a requested variable.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command="Store product price as item.price",
                situation="Evidence captured a runtime value for a requested variable.",
                reason="The command stores a description instead of the captured value.",
            ),
        ),
    ),
    CommandDoc(
        name="validate",
        syntax="Validate <subject> <state>",
        example="Validate checkout screen is visible",
        purpose="Assert a meaningful UI or data state.",
        rules=(
            "Subject must be a meaningful state or object proven by evidence.",
            "Do not validate generic controls unless the control itself is the user-level state.",
        ),
        examples=(
            CommandExample(
                kind=AuthoringExampleKind.PREFERRED,
                command="Validate phone number input field is visible",
                reason="The assertion names a meaningful screen state.",
                situation="A destination screen exposes a distinctive input field.",
            ),
            CommandExample(
                kind=AuthoringExampleKind.AVOID,
                command="Validate dismissal option is visible",
                situation="Only a generic dismissal option was tapped during an overlay.",
                reason="The assertion repeats a generic control instead of the user-level state.",
            ),
        ),
    ),
    CommandDoc(
        name="branch",
        syntax="IF <condition> { <commands> }",
        purpose="Represent a recorded conditional branch.",
        example="IF Permission dialog is visible { Tap on Allow button }",
        rules=(
            "Condition must be recorded in evidence.",
            "Branch body must contain only commands that ran under that condition.",
        ),
    ),
    CommandDoc(
        name="back",
        syntax="PRESS_DEVICE_BACK_BUTTON",
        example="PRESS_DEVICE_BACK_BUTTON",
        purpose="Press system device back button.",
        rules=("Use only for a recorded system-back action.",),
    ),
)
