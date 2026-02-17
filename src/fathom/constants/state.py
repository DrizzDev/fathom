"""
Constants for graph state keys.
"""

from enum import StrEnum


class StateKey(StrEnum):
    """Keys for IntentGraphState dictionary."""
    
    # Configuration
    INTENT = "intent"
    MAX_STEPS = "max_steps"
    USE_XML = "use_xml"
    
    # Execution
    STEP_NUMBER = "step_number"
    IS_COMPLETE = "is_complete"
    COMPLETION_REASON = "completion_reason"
    SHOULD_RETRY = "should_retry"
    INJECTED_CONTEXT = "injected_context"
    
    # Artefacts
    CAPTURE = "capture"
    SCREEN_STATE = "screen_state"
    IS_NEW_SCREEN = "is_new_screen"
    XML_CONTENT = "xml_content"
    ELEMENTS = "elements"
    
    # Analysis
    ANALYSIS = "analysis"
    PLAN = "plan"
    PLANNED_STEP = "planned_step"
    
    # Execution Result
    STEP_RESULT = "step_result"
    
    # Metrics
    GROUNDING_DURATION = "grounding_duration"
    ANALYSIS_DURATION = "analysis_duration"
    EXECUTION_DURATION = "execution_duration"
