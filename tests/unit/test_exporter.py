from fathom.services.exporter import ScriptExporter


def test_export_validation_event_uses_if_validate_for_transient_screen():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "wait",
            "target": "UI Element",
            "rationale": "Wait for loading spinner to finish before proceeding",
            "screen_changed": False,
        }
    ]

    script = ScriptExporter.export(step_results=steps, goal_state="")

    assert "IF Transient screen is visible { Validate app to finish loading }" in script


def test_export_validation_event_uses_if_validate_for_blocker_screen():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "wait",
            "target": "Permission dialog",
            "rationale": "Check if permission popup is visible before dismissing",
            "screen_changed": False,
        }
    ]

    script = ScriptExporter.export(step_results=steps, goal_state="")

    assert "IF Blocker prompt is visible { Validate Permission dialog }" in script


def test_export_non_validation_event_keeps_standard_if_structure():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "tap",
            "target": "Continue button",
            "condition": "Offer popup is visible",
            "rationale": "Dismiss optional popup",
            "screen_changed": True,
        }
    ]

    script = ScriptExporter.export(step_results=steps, goal_state="")

    assert "IF Offer popup is visible {" in script
    assert "    Validate Continue button is visible" in script
    assert "    Tap on Continue button" in script


def test_export_overlay_dismissal_infers_if_condition_from_rationale():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "tap",
            "target": "Got It! button",
            "rationale": "Dismissing the promotional overlay to clear the screen",
            "screen_changed": True,
        }
    ]

    script = ScriptExporter.export(step_results=steps, goal_state="")

    assert "IF Promotional overlay is visible {" in script
    assert "    Tap on Got It! button" in script


def test_export_validation_uses_specific_intent_subject_for_generic_target():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "validate",
            "target": "requested validation condition",
            "rationale": "Validation requested by user",
            "screen_changed": False,
        }
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent="validate that a lemon item with a price is visible in the search results",
    )

    assert "Validate that a lemon item with a price is visible in the search results" in script


def test_export_final_line_uses_intent_validation_subject():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "tap",
            "target": "Search button",
            "rationale": "Open results",
            "screen_changed": True,
        }
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent="open app then validate that the user is logged in",
    )

    assert "Validate that the user is logged in" in script


def test_export_validation_uses_multiple_intent_subjects_in_order():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "validate",
            "target": "profile icon",
            "rationale": "Check login state",
            "screen_changed": False,
        },
        {
            "step_number": 2,
            "event_type": "validation",
            "action_type": "validate",
            "target": "first search result",
            "rationale": "Check final result",
            "screen_changed": False,
        },
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent=(
            "Open app, validate that the user is logged in, then search lemon and "
            "validate that a lemon item with a price is visible in the search results"
        ),
    )

    assert "Validate that the user is logged in" in script
    assert "Validate that a lemon item with a price is visible in the search results" in script


def test_export_does_not_duplicate_final_validation_line():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "validate",
            "target": "requested validation condition",
            "rationale": "Final validation",
            "screen_changed": False,
        }
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent="validate that the user is logged in",
    )

    assert script.count("Validate that the user is logged in") == 1


def test_export_wraps_validation_after_wait_in_same_if():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "wait",
            "target": "the first search result",
            "rationale": "Wait for the first search result",
            "screen_changed": False,
        },
        {
            "step_number": 2,
            "event_type": "validation",
            "action_type": "validate",
            "target": "the first search result",
            "rationale": "Confirm result is shown",
            "screen_changed": False,
        },
    ]

    script = ScriptExporter.export(step_results=steps, intent="")

    assert "IF search results are still loading {" in script
    assert "    Wait for the first search result" in script
    assert "    Validate the first search result" in script
    assert (
        "IF the first search result is visible { Validate the first search result }" not in script
    )


def test_export_does_not_prepend_login_validation_precondition():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "tap",
            "target": "search bar",
            "rationale": "Start searching",
            "screen_changed": True,
        },
        {
            "step_number": 2,
            "event_type": "validation",
            "action_type": "validate",
            "target": "profile icon",
            "rationale": "Validate login state",
            "screen_changed": False,
        },
    ]

    script = ScriptExporter.export(
        step_results=steps,
        package_name="com.instacart.client",
        intent="Open app, validate that the user is logged in, then search for lemon",
    )

    lines = [line.strip() for line in script.splitlines() if line.strip()]
    assert lines[0] != "Validate that the user is logged in"
    assert lines[-1] == "Validate that the user is logged in"


def test_export_keeps_final_intent_validation_at_end_without_overlay_repetition():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "validate",
            "target": "Got It! button",
            "rationale": "Confirm overlay close control is visible",
            "screen_changed": False,
        },
        {
            "step_number": 2,
            "event_type": "action",
            "action_type": "tap",
            "target": "Got It! button",
            "rationale": "Dismissing the promotional overlay",
            "condition": "Promotional overlay is visible",
            "screen_changed": True,
        },
        {
            "step_number": 3,
            "event_type": "action",
            "action_type": "tap",
            "target": "search bar",
            "rationale": "Tap search field",
            "screen_changed": True,
        },
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent=(
            "open app, validate that the user is logged in, then search lemon and "
            "validate that you see lemon as an item with a price on it"
        ),
    )

    lines = [line.rstrip() for line in script.splitlines() if line.strip()]
    assert "Validate that the user is logged in" not in lines
    assert "Validate that you see lemon as an item with a price on it" in lines
    assert lines[-1] == "Validate that you see lemon as an item with a price on it"
    assert script.count("IF Promotional overlay is visible {") == 1
    assert "IF Promotional overlay is visible { Validate search bar is visible }" not in script


def test_export_does_not_wrap_final_intent_validation_with_previous_wait_condition():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "wait",
            "target": "loading indicator",
            "rationale": "Wait for loading indicator",
            "condition": "loading indicator is visible",
            "screen_changed": False,
        },
        {
            "step_number": 2,
            "event_type": "validation",
            "action_type": "validate",
            "target": "requested validation condition",
            "rationale": "Final user-requested validation",
            "screen_changed": False,
        },
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent="validate that you see lemon as an item with a price on it",
    )

    assert (
        "IF loading indicator is visible { Validate that you see lemon as an item with a price on it }"
        not in script
    )
    assert "Validate that you see lemon as an item with a price on it" in script


def test_export_frames_app_loading_wait_condition_more_clearly():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "wait",
            "target": "UI Element",
            "rationale": "Wait for app splash to load main interface",
            "screen_changed": False,
        }
    ]

    script = ScriptExporter.export(step_results=steps, intent="")

    assert "IF the app is still loading {" in script
    assert "    Wait for the app to finish loading" in script


def test_export_normalizes_pre_filled_wait_conditions_for_clarity():
    steps = [
        {
            "step_number": 1,
            "event_type": "action",
            "action_type": "wait",
            "target": "app to finish loading",
            "rationale": "Waiting for splash screen",
            "condition": "app to finish loading is visible",
            "screen_changed": False,
        },
        {
            "step_number": 2,
            "event_type": "action",
            "action_type": "wait",
            "target": "the first search result",
            "rationale": "Waiting for search results to appear",
            "condition": "the first search result is visible",
            "screen_changed": False,
        },
    ]

    script = ScriptExporter.export(step_results=steps, intent="")

    assert "IF the app is still loading {" in script
    assert "IF search results are still loading {" in script


def test_export_handles_noisy_intent_validation_subject_with_clean_grammar():
    steps = [
        {
            "step_number": 1,
            "event_type": "validation",
            "action_type": "validate",
            "target": "requested validation condition",
            "rationale": "Final validation",
            "screen_changed": False,
        }
    ]

    script = ScriptExporter.export(
        step_results=steps,
        intent="validate   that   user is   logged in  ",
    )

    assert "Validate that user is logged in" in script
