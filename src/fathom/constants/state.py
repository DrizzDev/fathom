from enum import StrEnum


class StateKey(StrEnum):
    """
    Keys for IntentGraphState dictionary.
    """

    # Configuration
    INTENT = "intent"
    USE_XML = "use_xml"
    MAX_STEPS = "max_steps"

    # Execution
    STEP_NUMBER = "step_number"
    IS_COMPLETE = "is_complete"
    COMPLETION_REASON = "completion_reason"

    SHOULD_RETRY = "should_retry"
    INJECTED_CONTEXT = "injected_context"

    # Artifacts
    CAPTURE = "capture"
    SCREEN_STATE = "screen_state"
    IS_NEW_SCREEN = "is_new_screen"

    ELEMENTS = "elements"
    XML_CONTENT = "xml_content"

    # Analysis
    PLAN = "plan"
    ANALYSIS = "analysis"
    PLANNED_STEP = "planned_step"

    # Execution Result
    STEP_RESULT = "step_result"

    # Metrics
    ANALYSIS_DURATION = "analysis_duration"
    GROUNDING_DURATION = "grounding_duration"
    EXECUTION_DURATION = "execution_duration"
