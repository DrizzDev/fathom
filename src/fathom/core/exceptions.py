"""Domain-specific exceptions for Fathom."""


class FathomError(Exception):
    """Base exception for all Fathom errors."""


class ExecutionError(FathomError):
    """Execution phase failed."""


class ConfigurationError(FathomError):
    """Invalid configuration."""


class StrategyError(FathomError):
    """Strategy execution failed."""


class PortError(FathomError):
    """Port communication failed."""
