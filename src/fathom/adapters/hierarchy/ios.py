from __future__ import annotations

from typing import Optional

from fathom.adapters.ios.gateway import IOSAutomationGateway
from fathom.constants.platform import IOSAutomationBackend
from fathom.core.exceptions import DeviceError
from fathom.interfaces.factory import HierarchyFactoryPort
from fathom.interfaces.hierarchy import HierarchyPort
from fathom.schemas.configuration import IOSConfiguration


class UnavailableHierarchyAdapter(HierarchyPort):
    """
    Hierarchy adapter used when hierarchy extraction is unavailable.
    """

    def __init__(self, *, reason: str) -> None:
        self.__reason = reason

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Raise deterministic error for unavailable hierarchy extraction.
        """

        raise DeviceError(self.__reason)


class IOSWebDriverAgentHierarchyAdapter(HierarchyPort):
    """
    iOS hierarchy extraction via WebDriverAgent session source endpoints.
    """

    def __init__(self, *, configuration: IOSConfiguration) -> None:
        self.__configuration = configuration
        self.__client = IOSAutomationGateway(configuration=configuration)

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Extract hierarchy XML by creating a short-lived WebDriverAgent session.
        """

        try:
            return await self.__fetch_source()
        except Exception as exception:
            raise DeviceError(
                f"Dump hierarchy: failed to fetch iOS hierarchy XML: {exception}"
            ) from exception

    async def __fetch_source(self) -> str:
        """
        Fetch hierarchy source using preferred and fallback endpoints.
        """

        return await self.__client.dump_source()


class IOSHierarchyAdapterFactory(HierarchyFactoryPort):
    """
    Factory for iOS hierarchy adapters based on iOS automation backend.
    """

    def create(self, *, configuration: IOSConfiguration) -> HierarchyPort:
        """
        Build hierarchy adapter for the provided iOS configuration.
        """

        if configuration.automation_backend == IOSAutomationBackend.APPIUM:
            return UnavailableHierarchyAdapter(
                reason="APPIUM backend is not supported for local iOS adapter hierarchy."
            )

        if configuration.automation_backend in {
            IOSAutomationBackend.XCUITEST,
            IOSAutomationBackend.WEBDRIVER_AGENT,
        }:
            return IOSWebDriverAgentHierarchyAdapter(configuration=configuration)

        if configuration.automation_backend == IOSAutomationBackend.XCRUN_SIMCTL:
            return UnavailableHierarchyAdapter(
                reason=(
                    "xcrun simctl does not currently expose a native XML hierarchy dump. "
                    "Use XCUITEST/WEBDRIVER_AGENT backend for hierarchy extraction."
                )
            )

        return UnavailableHierarchyAdapter(
            reason=(
                "No hierarchy adapter is configured for automation backend "
                f"{configuration.automation_backend.value}."
            )
        )
