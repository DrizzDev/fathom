from __future__ import annotations

import unittest
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.runtime.inspection import RuntimeConfigurationInspector


class _StubLLM:
    """
    Stand-in port object — only the class identity is logged.
    """


class _StubDevice:
    """
    Stand-in port object — only the class identity is logged.
    """


class _StubPathManager:
    """
    Minimal duck-type carrying the attributes the inspector projects.
    """

    base_path = "/tmp/fathom"
    memory_path = "/tmp/fathom/memory"
    output_path = "/tmp/fathom/output"


class _StubConfiguration(BaseModel):
    """
    Embeds clearly-sensitive fields so the redaction logic can be observed.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(default="vision-478905")
    api_key: str = Field(default="abcdef0123456789")
    credentials: Optional[str] = Field(default="{json: ...}")
    nested: dict = Field(
        default_factory=lambda: {
            "service_account": {"private_key": "-----BEGIN PRIVATE KEY----- xxx"},
            "max_steps": 10,
        }
    )
    tokens: Tuple[str, ...] = Field(default=("aaa-bearer", "bbb-bearer"))


_StubDevice.configuration = _StubConfiguration()


class RuntimeConfigurationInspectorTest(unittest.TestCase):
    """
    Pins the inspector's redaction + structural projection.
    """

    def test_port_instances_are_replaced_with_qualified_class_names(self) -> None:
        """
        The ports dict must surface ``module.ClassName`` strings, not instances.
        """

        snapshot = RuntimeConfigurationInspector().project(
            ports={"llm": _StubLLM(), "device": _StubDevice()},
            configuration=None,
            realignment=None,
            path_manager=None,
        )

        self.assertEqual(
            snapshot["ports"]["llm"],
            f"{_StubLLM.__module__}.{_StubLLM.__name__}",
        )
        self.assertEqual(
            snapshot["ports"]["device"],
            f"{_StubDevice.__module__}.{_StubDevice.__name__}",
        )

    def test_sensitive_keys_are_redacted(self) -> None:
        """
        Top-level keys that match the sensitive token list are redacted.
        """

        snapshot = RuntimeConfigurationInspector().project(
            ports={},
            configuration=_StubConfiguration(),
            realignment=None,
            path_manager=None,
        )
        config = snapshot["configuration"]

        self.assertEqual(config["project_id"], "vision-478905")
        self.assertEqual(config["api_key"], "[REDACTED]")
        self.assertEqual(config["credentials"], "[REDACTED]")
        self.assertEqual(config["tokens"], ["[REDACTED]", "[REDACTED]"])

    def test_nested_sensitive_subtrees_are_fully_redacted(self) -> None:
        """
        Every leaf reached under a sensitive key is redacted; non-sensitive
        siblings keep their original values.
        """

        snapshot = RuntimeConfigurationInspector().project(
            ports={},
            configuration=_StubConfiguration(),
            realignment=None,
            path_manager=None,
        )
        nested = snapshot["configuration"]["nested"]

        self.assertEqual(nested["service_account"]["private_key"], "[REDACTED]")
        self.assertEqual(nested["max_steps"], 10)

    def test_pem_like_string_values_are_redacted_even_under_safe_keys(self) -> None:
        """
        A value starting with PEM markers is redacted regardless of its key.
        """

        inspector = RuntimeConfigurationInspector()
        snapshot = inspector.project(
            ports={},
            configuration=_StubConfiguration(
                nested={
                    "innocuous_field": "-----BEGIN PRIVATE KEY----- payload",
                    "max_steps": 5,
                },
            ),
            realignment=None,
            path_manager=None,
        )

        nested = snapshot["configuration"]["nested"]
        self.assertEqual(nested["innocuous_field"], "[REDACTED]")
        self.assertEqual(nested["max_steps"], 5)

    def test_path_manager_exposes_only_directory_roots(self) -> None:
        """
        The projected paths section surfaces only the path-manager roots.
        """

        snapshot = RuntimeConfigurationInspector().project(
            ports={},
            configuration=None,
            realignment=None,
            path_manager=_StubPathManager(),
        )

        self.assertEqual(snapshot["paths"]["base_path"], "/tmp/fathom")
        self.assertEqual(snapshot["paths"]["memory_path"], "/tmp/fathom/memory")
        self.assertEqual(snapshot["paths"]["output_path"], "/tmp/fathom/output")

    def test_port_configuration_is_projected_and_redacted(self) -> None:
        """
        Port runtime configuration is logged separately from port identity.
        """

        snapshot = RuntimeConfigurationInspector().project(
            ports={"device": _StubDevice()},
            configuration=None,
            realignment=None,
            path_manager=None,
        )

        self.assertIn("device", snapshot["port_configuration"])
        device = snapshot["port_configuration"]["device"]
        self.assertEqual(device["project_id"], "vision-478905")
        self.assertEqual(device["api_key"], "[REDACTED]")

    def test_effective_device_configuration_overrides_default_device_snapshot(self) -> None:
        """
        Logged configuration.device reflects the wired device configuration, not schema defaults.
        """

        snapshot = RuntimeConfigurationInspector().project(
            ports={"device": _StubDevice()},
            configuration=_StubConfiguration(),
            realignment=None,
            path_manager=None,
        )

        self.assertEqual(
            snapshot["configuration"]["device"]["project_id"],
            "vision-478905",
        )
        self.assertEqual(
            snapshot["configuration"]["device"]["api_key"],
            "[REDACTED]",
        )
