from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.interfaces.lifecycle import RunnerLifecycle
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort


class QualifierComposition(BaseModel):
    """
    Value object returned by the qualifier composer.

    Bundles the qualifier port with the infrastructure resources the composer created; the composition
    root owns those and must close them when the run finishes — the qualifier itself never owns its LLM.
    ``resources`` is a tuple so the ownership view is genuinely immutable, not just reassignment-frozen.
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

    Pairs the runner with the resources needing explicit teardown, keeping lifecycle bookkeeping in a
    typed object the composition root can drain. ``runner`` is typed against RunnerLifecycle — a
    structural protocol of just cleanup() and cancel(), all the composition root needs; richer call
    sites keep their own reference to the concrete FathomRunner. ``resources`` is a tuple so the
    ownership view is genuinely immutable.
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
