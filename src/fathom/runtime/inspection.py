"""
Runtime inspection helpers — non-leaking introspection for wired Fathom runners.

Used at runner construction time to emit a single structured log line
summarising every effective configuration. Sensitive material
(credentials, API keys, service-account material, bearer tokens) is
masked. The output is a plain :class:`dict` so it slots into the
``extra=`` payload of a structured logger.
"""

from __future__ import annotations

from typing import Any, Dict, Final, Optional, Tuple

from pydantic import BaseModel


class RuntimeConfigurationInspector:
    """
    Project the runner's configuration tree into a redacted log-safe dict.
    """

    __REDACTED: Final[str] = "[REDACTED]"
    __SENSITIVE_KEY_TOKENS: Final[Tuple[str, ...]] = (
        "token",
        "secret",
        "auth",
        "bearer",
        "apikey",
        "api_key",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "private_key",
        "authorization",
        "client_secret",
        "service_account",
    )
    __SENSITIVE_VALUE_PREFIXES: Final[Tuple[str, ...]] = ("-----BEGIN",)

    def project(
        self,
        *,
        ports: Dict[str, Any],
        path_manager: Optional[Any],
        recovery: Optional[BaseModel],
        realignment: Optional[BaseModel],
        configuration: Optional[BaseModel],
    ) -> Dict[str, Any]:
        """
        Return one redacted dict carrying every config value worth logging.
        """

        return {
            "ports": self.__describe_ports(ports=ports),
            "paths": self.__describe_paths(path_manager=path_manager),
            "recovery": self.__redact(value=self.__as_dict(model=recovery)),
            "realignment": self.__redact(value=self.__as_dict(model=realignment)),
            "configuration": self.__redact(value=self.__as_dict(model=configuration)),
        }

    def __describe_ports(self, *, ports: Dict[str, Any]) -> Dict[str, str]:
        """
        Replace every port instance with its qualified class name.
        """

        return {name: self.__qualified_class_name(value=value) for name, value in ports.items()}

    def __describe_paths(self, *, path_manager: Any) -> Dict[str, Optional[str]]:
        """
        Surface the path manager's exposed roots without leaking file contents.
        """

        if path_manager is None:
            return {}

        attributes = (
            "xml_path",
            "base_path",
            "memory_path",
            "output_path",
            "screenshot_path",
        )
        return {
            name: self.__safe_str(value=getattr(path_manager, name, None)) for name in attributes
        }

    @staticmethod
    def __as_dict(*, model: Optional[BaseModel]) -> Dict[str, Any]:
        """
        Dump a Pydantic model to a plain dict, returning {} when absent.
        """

        if model is None:
            return {}

        try:
            return model.model_dump(mode="json")
        except Exception:
            return {"__unserializable__": True}

    @classmethod
    def __redact(cls, *, value: Any) -> Any:
        """
        Recursively replace sensitive values with the redaction sentinel.
        """

        return cls.__redact_node(node=value, sensitive=False)

    @classmethod
    def __redact_node(cls, *, node: Any, sensitive: bool) -> Any:
        """
        Walk one node, propagating the sensitive flag down a redacted subtree.
        """

        if isinstance(node, dict):
            return {
                key: cls.__redact_node(
                    node=child,
                    sensitive=sensitive or cls.__is_sensitive_key(key=key),
                )
                for key, child in node.items()
            }
        if isinstance(node, (list, tuple)):
            return [cls.__redact_node(node=item, sensitive=sensitive) for item in node]

        if sensitive:
            return cls.__REDACTED

        if isinstance(node, str) and cls.__looks_like_secret(value=node):
            return cls.__REDACTED

        return node

    @classmethod
    def __is_sensitive_key(cls, *, key: Any) -> bool:
        """
        Match a key against the sensitive-token allowlist (case-insensitive).
        """

        if not isinstance(key, str):
            return False

        normalized = key.lower()
        return any(token in normalized for token in cls.__SENSITIVE_KEY_TOKENS)

    @classmethod
    def __looks_like_secret(cls, *, value: str) -> bool:
        """
        Catch obviously-credential string values even when the key looks innocuous.
        """

        return any(value.startswith(prefix) for prefix in cls.__SENSITIVE_VALUE_PREFIXES)

    @staticmethod
    def __qualified_class_name(*, value: Any) -> str:
        """
        Return ``module.ClassName`` for ``value`` so the log identifies the adapter.
        """

        if value is None:
            return "None"

        cls = type(value)
        module = getattr(cls, "__module__", "<unknown>")

        return f"{module}.{cls.__name__}"

    @staticmethod
    def __safe_str(*, value: Any) -> Optional[str]:
        """
        Coerce a value to ``str`` defensively, returning ``None`` on failure.
        """

        if value is None:
            return None

        try:
            return str(value)
        except Exception:
            return None
