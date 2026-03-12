from typing import Callable, cast

from fathom.strategies.graph.intent.nodes import IntentNodeProvider


def _launcher_persistence_decision(
    provider: IntentNodeProvider, *, execution_activity: str, observed_activity: str
) -> bool:
    """Call the private launcher-persistence helper in a mypy-safe way."""

    decision_function = cast(
        "Callable[..., bool]",
        provider.__getattribute__("_IntentNodeProvider__should_skip_launcher_persistence"),
    )
    return decision_function(
        execution_activity=execution_activity,
        observed_activity=observed_activity,
    )


def test_should_skip_launcher_persistence_on_launcher_only_steps() -> None:
    """Skip persistence when execution both starts and ends on launcher packages."""

    provider = object.__new__(IntentNodeProvider)

    should_skip = _launcher_persistence_decision(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="com.google.android.apps.nexuslauncher",
        provider=provider,
    )

    assert should_skip is True


def test_should_persist_launcher_to_app_transition() -> None:
    """Persist steps that leave the launcher and open the requested app."""

    provider = object.__new__(IntentNodeProvider)

    should_skip = _launcher_persistence_decision(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="com.snabbit.customer",
        provider=provider,
    )

    assert should_skip is False


def test_should_skip_when_observed_package_is_unknown() -> None:
    """Keep launcher suppression when post-action package resolution is unavailable."""

    provider = object.__new__(IntentNodeProvider)

    should_skip = _launcher_persistence_decision(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="unknown",
        provider=provider,
    )

    assert should_skip is True
