from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

from pydantic import Field, model_validator

from fathom.constants.flow import (
    AssertionSource,
    CheckKind,
    IssueCode,
    LaunchProvenance,
    NodeKind,
    ScrollDirection,
)
from fathom.schemas.artifacts import StepArtifacts
from fathom.schemas.base import SealedModel
from fathom.schemas.steps import StepGoal


class Selector(SealedModel):
    """
    Vision-based reference to a UI target.
    """

    text: str = Field(min_length=1, description="Exact on-screen text or descriptor of the target.")
    position: Optional[str] = Field(
        default=None, description="Ordinal qualifier used when the target repeats, e.g. 'first'."
    )
    container: Optional[str] = Field(
        default=None, description="Section or container the target belongs to."
    )


class Check(SealedModel):
    """
    A single validation assertion.
    """

    kind: CheckKind = Field(description="Category of the assertion.")
    subject: str = Field(min_length=1, description="What the assertion is about.")


class Guard(SealedModel):
    """
    Visibility condition for a conditional branch, grounded in recorded evidence.
    """

    condition: str = Field(min_length=1, description="Recorded condition text used verbatim.")
    source_step: int = Field(ge=0, description="Step number the condition was recorded on.")


class Node(SealedModel):
    """
    Base for flow nodes carrying evidence provenance.
    """

    source_steps: Tuple[int, ...] = Field(
        min_length=1, description="Evidence step numbers this node was derived from."
    )


class LaunchNode(Node):
    """
    Launch the target application.
    """

    kind: Literal[NodeKind.LAUNCH] = NodeKind.LAUNCH
    package: str = Field(min_length=1, description="Target application package.")
    provenance: LaunchProvenance = Field(
        default=LaunchProvenance.SYNTHETIC_WARM_START,
        description="How this launch was synthesised.",
    )
    source_steps: Tuple[int, ...] = Field(
        default=(), description="Collapsed launcher step numbers grounding a launcher transition."
    )


class TapNode(Node):
    """
    Tap a UI target.
    """

    kind: Literal[NodeKind.TAP] = NodeKind.TAP
    selector: Selector = Field(description="Target to tap.")


class TypeNode(Node):
    """
    Type a value into a field.
    """

    kind: Literal[NodeKind.TYPE] = NodeKind.TYPE
    text: str = Field(min_length=1, description="Text value to enter.")
    field: Selector = Field(description="Field to type into.")


class ScrollNode(Node):
    """
    Scroll in a direction.
    """

    kind: Literal[NodeKind.SCROLL] = NodeKind.SCROLL
    direction: ScrollDirection = Field(description="Scroll direction.")
    container: Optional[str] = Field(default=None, description="Container scrolled inside.")
    percentage: Optional[int] = Field(
        default=None, ge=1, le=100, description="Fraction of the view to scroll by."
    )

    @model_validator(mode="after")
    def __single_variant(self) -> "ScrollNode":
        """
        Allow at most one scroll qualifier so the renderer never drops data.
        """

        if self.container is not None and self.percentage is not None:
            raise ValueError("Scroll allows at most one of container or percentage.")

        return self


class ScrollUntilNode(Node):
    """
    Scroll in a direction until a target is visible.
    """

    kind: Literal[NodeKind.SCROLL_UNTIL] = NodeKind.SCROLL_UNTIL
    direction: ScrollDirection = Field(description="Scroll direction.")
    target: str = Field(min_length=1, description="Target text to scroll until visible.")
    container: Optional[str] = Field(default=None, description="Container scrolled inside.")


class WaitNode(Node):
    """
    Wait for a duration in seconds or until a subject appears.
    """

    kind: Literal[NodeKind.WAIT] = NodeKind.WAIT
    subject: Optional[str] = Field(default=None, min_length=1, description="Subject to wait for.")
    duration: Optional[int] = Field(
        default=None, ge=0, description="Wait duration in whole seconds."
    )

    @model_validator(mode="after")
    def __require_one_form(self) -> "WaitNode":
        """
        Require at least one of duration or subject.
        """

        if self.duration is None and self.subject is None:
            raise ValueError("Wait node requires a duration or a subject.")

        return self


class BackNode(Node):
    """
    Press the device back button.
    """

    kind: Literal[NodeKind.BACK] = NodeKind.BACK


class KillNode(Node):
    """
    Force-close the active application.
    """

    kind: Literal[NodeKind.KILL] = NodeKind.KILL


class ClearNode(Node):
    """
    Clear the active application's data.
    """

    kind: Literal[NodeKind.CLEAR] = NodeKind.CLEAR


class MinimizeNode(Node):
    """
    Send the active application to the background.
    """

    kind: Literal[NodeKind.MINIMIZE] = NodeKind.MINIMIZE


class LocationNode(Node):
    """
    Set the device GPS coordinates.
    """

    kind: Literal[NodeKind.LOCATION] = NodeKind.LOCATION
    latitude: float = Field(description="Latitude in decimal degrees.")
    longitude: float = Field(description="Longitude in decimal degrees.")


class StoreNode(Node):
    """
    Store a captured value under a variable name.
    """

    kind: Literal[NodeKind.STORE] = NodeKind.STORE
    value: str = Field(min_length=1, description="Captured value to store.")
    name: str = Field(min_length=1, description="Variable name to store under.")


class CheckNode(Node):
    """
    Assert one or more UI states.
    """

    kind: Literal[NodeKind.CHECK] = NodeKind.CHECK
    checks: Tuple[Check, ...] = Field(min_length=1, description="Assertions to verify.")
    assertion_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Completion assertion identifiers grounding this validation.",
    )


class MapNode(Node):
    """
    Tap a target on a map or canvas surface.
    """

    kind: Literal[NodeKind.MAP] = NodeKind.MAP
    selector: Selector = Field(description="Map target to tap.")


# Plain (non-discriminated) unions: each node carries a Literal `kind`, so Pydantic
# smart-union matching stays correct, while the JSON schema emits `anyOf` instead of
# `oneOf`+`discriminator` — the form the Gemini/Vertex structured-output schema accepts.
LeafNode = Union[
    MapNode,
    TapNode,
    TypeNode,
    WaitNode,
    BackNode,
    KillNode,
    ClearNode,
    StoreNode,
    CheckNode,
    ScrollNode,
    LaunchNode,
    MinimizeNode,
    LocationNode,
    ScrollUntilNode,
]


class BranchNode(Node):
    """
    Conditionally execute a body of leaf nodes under a visibility guard.
    """

    kind: Literal[NodeKind.BRANCH] = NodeKind.BRANCH
    guard: Guard = Field(description="Visibility condition for the branch.")
    body: Tuple[LeafNode, ...] = Field(min_length=1, description="Leaf nodes run when true.")


FlowNode = Union[
    TapNode,
    MapNode,
    TypeNode,
    WaitNode,
    BackNode,
    KillNode,
    ClearNode,
    StoreNode,
    CheckNode,
    LaunchNode,
    BranchNode,
    ScrollNode,
    MinimizeNode,
    LocationNode,
    ScrollUntilNode,
]


class Flow(SealedModel):
    """
    Target-neutral representation of a generated automation script.
    """

    intent: str = Field(description="User intent the script fulfils.")
    package: str = Field(description="Target application package.")
    nodes: Tuple[FlowNode, ...] = Field(default_factory=tuple, description="Ordered flow nodes.")
    partial: bool = Field(
        default=False, description="True when the run did not complete and the script needs review."
    )


class Issue(SealedModel):
    """
    A single problem found by a validation gate.
    """

    code: IssueCode = Field(description="Identifier for the kind of problem.")
    message: str = Field(description="Actionable description of the problem.")
    node_index: Optional[int] = Field(
        default=None, description="Index of the offending node when applicable."
    )


class Report(SealedModel):
    """
    Outcome of a validation gate.
    """

    issues: Tuple[Issue, ...] = Field(
        default_factory=tuple, description="Problems found; empty when valid."
    )

    @property
    def ok(self) -> bool:
        """
        Return whether the validated subject passed with no issues.
        """

        return not self.issues


class StepTarget(SealedModel):
    """
    The UI target an action addressed.
    """

    export: Optional[str] = Field(default=None, description="Canonical phrase for the target.")

    element: Optional[str] = Field(default=None, description="Element role of the target.")
    scroll: Optional[str] = Field(default=None, description="Element being scrolled toward.")
    name: Optional[str] = Field(default=None, description="Raw on-screen name of the target.")

    generalized: Optional[str] = Field(default=None, description="Generalized phrase when dynamic.")
    positional: bool = Field(
        default=False, description="Whether the target is an ordinal reference."
    )


class StepWait(SealedModel):
    """
    What a step waited for.
    """

    subject: Optional[str] = Field(default=None, description="Subject being waited for.")
    pattern: Optional[str] = Field(default=None, description="Wait category.")


class StepGuard(SealedModel):
    """
    The condition under which a step executed.
    """

    condition: Optional[str] = Field(default=None, description="Recorded condition or marker.")
    conditional: bool = Field(default=False, description="Whether the step executed under a guard.")

    kind: Optional[str] = Field(default=None, description="Conditional category.")
    overlay: bool = Field(
        default=False, description="Whether the step handled an overlay or popup."
    )


class StepOutcome(SealedModel):
    """
    The recorded result of executing a step.
    """

    success: bool = Field(default=True, description="Whether execution reported success.")
    changed: bool = Field(default=False, description="Whether the screen changed after the step.")

    duration: Optional[int] = Field(
        default=None, ge=0, description="Execution duration in milliseconds."
    )


class StepLaunch(SealedModel):
    """
    A deterministically synthesized app launch the model must render as a LaunchNode.
    """

    package: str = Field(min_length=1, description="Real app package to launch.")
    provenance: LaunchProvenance = Field(description="How the launch was synthesized.")

    source_steps: Tuple[int, ...] = Field(
        default_factory=tuple, description="Collapsed launcher step numbers grounding the launch."
    )


class StepCapture(SealedModel):
    """
    A value a STORE step requested and the outcome of capturing it, exposed to script generation.
    """

    name: str = Field(description="Variable name the value is stored under.")
    subject: str = Field(description="What the intent asked to capture.")
    success: bool = Field(description="Whether the capture succeeded at record time.")

    value: Optional[str] = Field(
        default=None,
        description="Runtime captured value rendered by Store when capture succeeded.",
    )
    reason: Optional[str] = Field(
        default=None, description="Failure reason when the capture failed."
    )


class CompletionAssertion(SealedModel):
    """
    Terminal assertion proven by execution verification and available to script authoring.
    """

    id: str = Field(min_length=1, description="Stable assertion identifier within the execution.")
    kind: CheckKind = Field(description="Validation kind proven by the verifier.")
    source: AssertionSource = Field(description="System component that produced the assertion.")
    subject: str = Field(min_length=1, description="Visible state or data subject that was proven.")

    reason: Optional[str] = Field(default=None, description="Verifier rationale for the assertion.")
    step_index: Optional[int] = Field(
        default=None, ge=0, description="Execution step after which the assertion was proven."
    )
    artifacts: Tuple[str, ...] = Field(
        default_factory=tuple, description="Artifacts inspected to prove the assertion."
    )


class EvidenceStep(SealedModel):
    """
    One recorded execution step exposed to script generation.
    """

    action: str = Field(description="Recorded action type.")
    index: int = Field(ge=0, description="Step sequence number.")
    event: str = Field(description="High-level event category, e.g. action, validation, or launch.")

    launch: Optional[StepLaunch] = Field(
        default=None, description="Synthesized launch to render as a LaunchNode, when event=launch."
    )
    text: Optional[str] = Field(default=None, description="Typed text content.")

    rationale: Optional[str] = Field(default=None, description="Planner reasoning for the step.")
    observation: Optional[str] = Field(
        default=None, description="Planner observation of the screen."
    )
    goal: Optional[StepGoal] = Field(
        default=None, description="Compact sub-goal context active for this step."
    )

    screenshot: Optional[str] = Field(default=None, description="Reference to the step screenshot.")
    artifacts: Optional[StepArtifacts] = Field(
        default=None, description="Structured artifacts captured during the step."
    )
    target: StepTarget = Field(default_factory=StepTarget, description="Target the step addressed.")

    wait: StepWait = Field(default_factory=StepWait, description="What the step waited for.")
    guard: StepGuard = Field(default_factory=StepGuard, description="Condition the step ran under.")

    outcome: StepOutcome = Field(
        default_factory=StepOutcome, description="Recorded execution result."
    )

    capture: Optional[StepCapture] = Field(
        default=None,
        description="The STORE capture request and outcome, when this step stored a value.",
    )


class Evidence(SealedModel):
    """
    Full per-run evidence aggregate consumed by script generation.
    """

    intent: str = Field(description="User intent for the run.")
    goal: str = Field(description="Goal-state description for the run.")

    package: str = Field(description="Target application package.")

    steps: Tuple[EvidenceStep, ...] = Field(
        default_factory=tuple, description="Ordered recorded steps."
    )
    artifacts: Tuple[str, ...] = Field(
        default_factory=tuple, description="Run-level artifact references."
    )
    assertions: Tuple[CompletionAssertion, ...] = Field(
        default_factory=tuple, description="Terminal assertions proven outside normal steps."
    )
    partial: bool = Field(
        default=False, description="True when no successful goal validation was recorded."
    )
    discarded: Tuple[int, ...] = Field(
        default_factory=tuple, description="Step numbers dropped during distillation."
    )
    reason: Optional[str] = Field(
        default=None, description="Why the run was distilled to partial, when it was."
    )

    @model_validator(mode="after")
    def __unique_step_numbers(self) -> "Evidence":
        """
        Reject duplicate recorded step numbers so the step lookup is lossless.
        """

        numbers = [step.index for step in self.steps if step.launch is None]

        if len(numbers) != len(set(numbers)):
            raise ValueError("Evidence step numbers must be unique.")

        return self


class RunObjective(SealedModel):
    """
    The user objective for a run, carried alongside the run identifier to evidence assembly.
    """

    intent: str = Field(min_length=1, description="User intent for the run.")
    goal: str = Field(default="", description="Goal-state description for the run.")

    package: str = Field(min_length=1, description="Target application package.")
