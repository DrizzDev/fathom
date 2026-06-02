from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.interfaces.lifecycle import RunnerLifecycle
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort


class QualifierComposition(BaseModel):
    """
    Value object returned by the qualifier composer.

    Bundles the qualifier port with any infrastructure resources the composer created on the caller's behalf.
    The composition root owns the resources and must close them when the run finishes — the qualifier itself never owns its LLM.

    `resources` is a tuple so the ownership view is genuinely immutable;
    frozen=True alone blocks field reassignment but would still let callers mutate a list via append/extend.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    qualifier: IntentQualifierPort = Field(
        description="Qualifier port composed for the run.",
    )
    resources: Tuple[LLMPort, ...] = Field(
        default_factory=tuple,
        description=(
            "Runtime resources created by the composer and owned by the composition "
            "root. The runtime must call cleanup() on each entry after the runner completes."
        ),
    )


class RunnerComposition(BaseModel):
    """
    Value object returned by the activity / executor builder.

    Pairs the runner with the resources that need explicit teardown alongside
    the runner itself. Keeps lifecycle bookkeeping out of the runner contract
    and inside a typed object the composition root can drain.

    The `runner` field is typed against RunnerLifecycle — a small structural
    protocol exposing just cleanup() and cancel(). That's all the composition
    root needs from this object; richer call sites (activity / CLI executor)
    keep their own typed reference to the concrete FathomRunner.
    `resources` is a tuple so the ownership view is genuinely immutable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    runner: RunnerLifecycle = Field(
        description=(
            "Runner composed for execution. Typed against the lifecycle protocol; "
            "concrete type is FathomRunner."
        ),
    )
    resources: Tuple[LLMPort, ...] = Field(
        default_factory=tuple,
        description=(
            "Runtime resources owned by the composition root (e.g. dedicated "
            "qualifier LLM). Closed by the composition root after runner cleanup."
        ),
    )
