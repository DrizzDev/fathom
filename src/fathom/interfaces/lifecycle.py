from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunnerLifecycle(Protocol):
    """
    Minimum runtime lifecycle a composition root needs to manage a runner.

    Used by RunnerComposition.runner so the schema layer can stay decoupled
    from the concrete FathomRunner implementation while still expressing the
    cleanup / cancel contract the composition root relies on.

    Runtime callers that need richer surface (run_intent, run_exploration,
    device access) must keep the concrete FathomRunner reference at their
    own layer; this protocol intentionally exposes only lifecycle methods.
    """

    async def cleanup(self) -> None:
        """
        Release runner-owned resources. Must be safe to call exactly once.
        """

        ...

    def cancel(self) -> None:
        """
        Signal the runner to abort an in-progress execution synchronously.
        """

        ...
