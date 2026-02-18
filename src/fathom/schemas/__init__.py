from __future__ import annotations

from fathom.schemas.actions import Action, Bounds
from fathom.schemas.configuration import (
    ADBConfiguration,
    DeviceConfiguration,
    ExecutionConfiguration,
    ExplorationConfiguration,
    FathomConfiguration,
    IntentConfiguration,
    LLMConfiguration,
)
from fathom.schemas.exploration import ActionGenerator, ExplorationGraph, ScreenNode
from fathom.schemas.orchestration import (
    ExecutionContext,
    RunnerConfig,
    RunnerResult,
    StepContext,
)
from fathom.schemas.results import (
    ActionResult,
    AnalysisResult,
    ExecutionResult,
    ExplorationResult,
    IntentResult,
    PlanResult,
    StrategyResult,
    WorkflowResult,
)
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepRecord, StepResult
from fathom.schemas.ui import LabeledElement, UIBounds

# Backward compatibility aliases
ADBConfig = ADBConfiguration
ExecutionConfig = ExecutionConfiguration
ExplorationStrategyConfig = ExplorationConfiguration
FathomConfig = FathomConfiguration
GeminiConfig = LLMConfiguration
IntentStrategyConfig = IntentConfiguration
WorkflowConfig = ExecutionConfiguration

__all__ = [
    "ADBConfig",
    "ADBConfiguration",
    "Action",
    "ActionGenerator",
    "ActionResult",
    "AnalysisResult",
    "Bounds",
    "DeviceConfiguration",
    "ExecutionConfig",
    "ExecutionConfiguration",
    "ExecutionContext",
    "ExecutionResult",
    "ExplorationConfiguration",
    "ExplorationGraph",
    "ExplorationResult",
    "ExplorationStrategyConfig",
    "FathomConfig",
    "FathomConfiguration",
    "GeminiConfig",
    "IntentConfiguration",
    "IntentResult",
    "IntentStrategyConfig",
    "LabeledElement",
    "LLMConfiguration",
    "PlanResult",
    "RunnerConfig",
    "RunnerResult",
    "ScreenCapture",
    "ScreenNode",
    "ScreenState",
    "Step",
    "StepContext",
    "StepRecord",
    "StepResult",
    "StrategyResult",
    "UIBounds",
    "WorkflowConfig",
    "WorkflowResult",
]
