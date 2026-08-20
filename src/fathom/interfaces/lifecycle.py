from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunnerLifecycle(Protocol):
    """
    Lifecycle-only view of a runner: the cleanup and cancel a composition root drives.

    Lets RunnerComposition.runner stay decoupled from the concrete FathomRunner. Callers needing richer
    surface (run_intent, run_exploration, device access) must keep their own concrete FathomRunner
    reference; this protocol intentionally exposes only lifecycle methods.
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
