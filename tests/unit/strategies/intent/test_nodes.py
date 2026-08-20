from fathom.strategies.graph.intent.nodes.persistence import GraphStatePersistence


def test_should_skip_launcher_persistence_on_launcher_only_steps() -> None:
    """
    Skip persistence when execution both starts and ends on launcher packages.
    """

    should_skip = GraphStatePersistence.should_skip_launcher(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="com.google.android.apps.nexuslauncher",
    )

    assert should_skip is True


def test_should_persist_launcher_to_app_transition() -> None:
    """
    Persist steps that leave the launcher and open the requested app.
    """

    should_skip = GraphStatePersistence.should_skip_launcher(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="com.example.customer",
    )

    assert should_skip is False


def test_should_skip_when_observed_package_is_unknown() -> None:
    """
    Keep launcher suppression when post-action package resolution is unavailable.
    """

    should_skip = GraphStatePersistence.should_skip_launcher(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="unknown",
    )

    assert should_skip is True
