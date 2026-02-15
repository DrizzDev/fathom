"""Core exceptions for Fathom."""

from __future__ import annotations


class FathomCoreError(Exception):
    """Base exception for core errors."""

    pass


class StrategyError(FathomCoreError):
    """Exception raised by strategy execution."""

    pass


class ConfigurationError(FathomCoreError):
    """Exception raised for configuration errors."""

    pass


class ExecutionError(FathomCoreError):
    """Exception raised during execution."""

    pass


class PortError(FathomCoreError):
    """Exception raised by port operations."""

    pass
