# Configuration Fixes - Missing Schemas

## Issues Fixed

### 1. Missing Configuration Schemas
The new configuration schema was missing several classes that the old code depends on:
- `ADBCaptureConfig` - Used by `ADBCaptureTool`
- `HasherConfig` - Used by hybrid hasher
- `WorkflowConfig` - Used by workflows

### 2. Missing Core Exceptions
The `src/fathom/core/exceptions.py` file was empty, missing:
- `StrategyError` - Used by strategies
- `ConfigurationError` - Used by builder
- `ExecutionError` - Used by execution engine
- `PortError` - Used by ports
- `FathomCoreError` - Base exception

### 3. CLI Not Passing All Settings
The new CLI was only passing `api_key` and `model` to GeminiLLM, but it needs ALL settings including:
- `credentials_path` - For Google Cloud service account
- `project_id` - For Vertex AI
- `location` - For Vertex AI region

## What Was Added

### Configuration Schemas (`src/fathom/schemas/configuration.py`)

```python
class ADBCaptureConfig(BaseModel):
    """Configuration for ADB capture tool."""
    adb_path: str = "adb"
    timeout: float = 10.0
    use_hybrid_hash: bool = True
    device_serial: Optional[str] = None

class HasherConfig(BaseModel):
    """Configuration for hybrid hasher."""
    use_perceptual: bool = True
    use_structural: bool = True
    thumbnail_size: Tuple[int, int] = (8, 8)

class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""
    max_steps: int = 20
    step_timeout: float = 15.0
    total_timeout: float = 600.0
    checkpoint_interval: int = 5
    retry_limit: int = 3
    use_xml_bounding_boxes: bool = False
    package_name: Optional[str] = None
```

### Core Exceptions (`src/fathom/core/exceptions.py`)

```python
class FathomCoreError(Exception):
    """Base exception for core errors."""

class StrategyError(FathomCoreError):
    """Exception raised by strategy execution."""

class ConfigurationError(FathomCoreError):
    """Exception raised for configuration errors."""

class ExecutionError(FathomCoreError):
    """Exception raised during execution."""

class PortError(FathomCoreError):
    """Exception raised by port operations."""
```

### CLI Fix (`src/fathom/cli_new.py`)

Changed from:
```python
.llm(
    GeminiLLM(
        api_key=self.settings.gemini_api_key,
        model=self.settings.gemini_model,
    )
)
```

To:
```python
gemini_config = GeminiConfig(
    model=self.settings.gemini_model,
    api_key=self.settings.gemini_api_key,
    location=self.settings.vertex_location,
    project_id=self.settings.vertex_project_id,
    credentials_path=self.settings.google_application_credentials,
)

.llm(GeminiLLM(configuration=gemini_config))
```

## Why This Matters

1. **Backward Compatibility**: Old code imports these schemas, so they must exist
2. **Vertex AI Support**: Users with Google Cloud credentials can now use Vertex AI
3. **Complete Configuration**: All settings from FathomSettings are now passed through
4. **Proper Error Handling**: Core exceptions are now defined for proper error propagation

## Verification

```bash
# Test imports
python -c "from fathom.strategies.intent import IntentStrategy; print('✅ Success')"
python -c "from fathom.schemas.configuration import ADBCaptureConfig, HasherConfig, WorkflowConfig; print('✅ Success')"
python -c "from fathom.core.exceptions import StrategyError, ConfigurationError; print('✅ Success')"

# Test CLI
fathom --help
```

All imports should now work correctly and the CLI should be able to use Google Cloud credentials.
