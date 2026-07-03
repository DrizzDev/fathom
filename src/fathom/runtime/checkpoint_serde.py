"""
Fathom-side serde builder for LangGraph checkpoints with allow-listed Pydantic types.
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar, Dict, Tuple


class CheckpointSerdeFactory:
    """
    Build a LangGraph JsonPlusSerializer with the fathom-domain Pydantic type allow-list.
    """

    __ALLOWED_JSON_MODULES: ClassVar[Tuple[Tuple[str, str], ...]] = (
        ("fathom.constants", "ActionType"),
        ("fathom.constants.command", "CommandScopeKind"),
        ("fathom.constants.observation", "KeyboardVisibility"),
        ("fathom.constants.scroll", "ScrollEvidenceSource"),
        ("fathom.constants.storage", "StorageBackend"),
        ("fathom.constants.swipe", "AbortReason"),
        ("fathom.constants.tools", "DiagnosticSeverity"),
        ("fathom.constants.tools", "StateNamespace"),
        ("fathom.schemas.actions", "Action"),
        ("fathom.schemas.actions", "Bounds"),
        ("fathom.schemas.actions", "ExecutionRegion"),
        ("fathom.schemas.actions", "GesturePath"),
        ("fathom.schemas.actions", "InputContext"),
        ("fathom.schemas.actions", "CoordinateSystem"),
        ("fathom.schemas.actions", "CoordinateSource"),
        ("fathom.schemas.actions", "InputContextSource"),
        ("fathom.schemas.capture", "Capture"),
        ("fathom.schemas.capture", "CaptureRequest"),
        ("fathom.schemas.artifacts", "ScreenArtifact"),
        ("fathom.schemas.artifacts", "ScreenArtifactBundle"),
        ("fathom.schemas.artifacts", "StepArtifacts"),
        ("fathom.schemas.delta", "DeltaSignal"),
        ("fathom.schemas.effect", "ActionEffectSignalCounts"),
        ("fathom.schemas.execution", "ExecutionContext"),
        ("fathom.schemas.gemini_tools", "ExecuteAction"),
        ("fathom.schemas.gemini_tools", "GeminiBBox"),
        ("fathom.schemas.localization", "LocalizationCandidate"),
        ("fathom.schemas.localization", "LocalizationResult"),
        ("fathom.schemas.localization", "LocalizationStatus"),
        ("fathom.schemas.localization", "Point"),
        ("fathom.schemas.observation", "ElementRole"),
        ("fathom.schemas.observation", "ElementSource"),
        ("fathom.schemas.observation", "KeyboardObservation"),
        ("fathom.schemas.observation", "OverlayObservation"),
        ("fathom.schemas.observation", "PerceivedElement"),
        ("fathom.schemas.observation", "ScreenObservation"),
        ("fathom.schemas.observation", "ScrollRegion"),
        ("fathom.schemas.results", "AnalysisOutcome"),
        ("fathom.schemas.results", "AnalysisResult"),
        ("fathom.schemas.results", "ActionTraceAttempt"),
        ("fathom.schemas.results", "ActionTraceEvent"),
        ("fathom.schemas.results", "ExecutionResult"),
        ("fathom.schemas.results", "PlanResult"),
        ("fathom.schemas.results", "TraceEmission"),
        ("fathom.schemas.screens", "ScreenCapture"),
        ("fathom.schemas.screens", "ScreenChangeRegion"),
        ("fathom.schemas.screens", "ScreenDiff"),
        ("fathom.schemas.screens", "ScreenHashBundle"),
        ("fathom.schemas.screens", "ScreenScrollTranslation"),
        ("fathom.schemas.screens", "ScreenState"),
        ("fathom.schemas.swipe", "CandidateSequence"),
        ("fathom.schemas.swipe", "DeviceOutcome"),
        ("fathom.schemas.swipe", "SwipeAttempt"),
        ("fathom.schemas.swipe", "SwipeExecution"),
        ("fathom.schemas.swipe", "SwipeRejection"),
        ("fathom.schemas.swipe", "VisualOutcome"),
        ("fathom.schemas.steps", "Step"),
        ("fathom.schemas.steps", "StepGoal"),
        ("fathom.schemas.steps", "StepResult"),
        ("fathom.schemas.tools", "StateUpdate"),
        ("fathom.schemas.tools", "ToolArtifact"),
        ("fathom.schemas.tools", "ToolCommand"),
        ("fathom.schemas.tools", "ToolData"),
        ("fathom.schemas.tools", "ToolDiagnostic"),
        ("fathom.schemas.tools", "ToolResponse"),
    )

    @classmethod
    def allowed_json_modules(cls) -> Tuple[Tuple[str, str], ...]:
        """
        Allow-list of (module, qualname) pairs for JSON checkpoint deserialization.
        """

        return cls.__ALLOWED_JSON_MODULES

    @classmethod
    def allowed_msgpack_modules(cls) -> Tuple[Tuple[str, str], ...]:
        """
        Allow-list of (module, qualname) pairs for msgpack checkpoint deserialization.
        """

        return cls.__ALLOWED_JSON_MODULES

    @classmethod
    def build(cls) -> Any:
        """
        Construct a JsonPlusSerializer pre-configured with the fathom allow-list.
        """

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        configuration: Dict[str, Any] = {
            "allowed_json_modules": cls.__ALLOWED_JSON_MODULES,
        }
        signature = inspect.signature(JsonPlusSerializer)
        if "allowed_msgpack_modules" in signature.parameters:
            configuration["allowed_msgpack_modules"] = cls.__ALLOWED_JSON_MODULES

        return JsonPlusSerializer(**configuration)
