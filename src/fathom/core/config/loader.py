from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

from fathom.schemas.localization import EnsembleMemberName
from fathom.schemas.perception import (
    CvConfiguration,
    DocumentAiCredentials,
    IconConfiguration,
    JournalConfiguration,
    KeyboardConfiguration,
    LocalizationEnsembleConfiguration,
    OcrConfiguration,
    OverlayConfiguration,
    PerceptionConfiguration,
)
from fathom.settings.env import FathomSettings

logger = getLogger(__name__)


class RuntimeConfigLoader:
    """
    Translates flat :class:`FathomSettings` env-derived fields into the
    nested boot-time configuration models the runtime consumes.

    The loader does no env-var I/O of its own: every value comes from
    the already-validated settings object so there is exactly one place
    in the codebase that knows the FATHOM_* variable names.
    """

    def __init__(self, *, settings: Optional[FathomSettings] = None) -> None:
        """
        Initialize the loader against an optional settings object for tests.
        """

        self.__settings = settings if settings is not None else FathomSettings()

    def perception(self) -> PerceptionConfiguration:
        """
        Resolve the perception configuration from settings.
        """

        ocr_enabled = self.__settings.observation_ocr_enabled
        document_ai = self.__document_ai_credentials(required=ocr_enabled)

        return PerceptionConfiguration(
            ocr=OcrConfiguration(enabled=ocr_enabled, document_ai=document_ai),
            cv=CvConfiguration(enabled=self.__settings.observation_cv_enabled),
            icon=IconConfiguration(enabled=self.__settings.observation_icon_enabled),
            overlay=OverlayConfiguration(enabled=self.__settings.observation_overlay_enabled),
            keyboard=KeyboardConfiguration(
                enabled=self.__settings.observation_keyboard_enabled,
            ),
            journal=JournalConfiguration(local_enabled=self.__settings.journal_local_enabled),
        )

    def localization(self) -> LocalizationEnsembleConfiguration:
        """
        Resolve the ensemble vision-localizer configuration from settings.
        """

        if not self.__settings.ensemble_localizer_enabled:
            return LocalizationEnsembleConfiguration(enabled=False, members=())

        return LocalizationEnsembleConfiguration(enabled=True, members=self.__members())

    def __document_ai_credentials(self, *, required: bool) -> Optional[DocumentAiCredentials]:
        """
        Resolve Document AI credentials when OCR is enabled; otherwise return None.
        """

        project = (self.__settings.document_ai_project or "").strip()
        location = (self.__settings.document_ai_location or "").strip()
        processor = (self.__settings.document_ai_processor or "").strip()

        if not (project and location and processor):
            if required:
                logger.warning(
                    "OCR requested but Document AI credentials are not fully configured.",
                    extra={
                        "component": "core.config.loader",
                        "event": "config.documentai.missing",
                        "project.set": bool(project),
                        "location.set": bool(location),
                        "processor.set": bool(processor),
                    },
                )
            return None

        return DocumentAiCredentials(
            project=project,
            location=location,
            processor=processor,
            credentials=self.__google_credentials(),
        )

    def __google_credentials(self) -> Optional[Union[Dict[str, Any], str]]:
        """
        Return the same Google service-account material the Gemini adapter
        consumes. Inline JSON dict wins over the file path so a payload
        provided directly via ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` is
        honored over an on-disk key file.
        """

        return (
            self.__settings.google_credentials_dict
            or self.__settings.google_application_credentials
        )

    def __members(self) -> Tuple[EnsembleMemberName, ...]:
        """
        Parse the comma-separated ensemble-member list and validate every name.
        """

        raw = (self.__settings.ensemble_localizer_members or "").strip()
        if not raw:
            return ()

        candidates = tuple(member.strip() for member in raw.split(",") if member.strip())
        try:
            return tuple(EnsembleMemberName(name) for name in candidates)
        except ValueError as exception:
            supported = tuple(member.value for member in EnsembleMemberName)
            raise ValueError(
                f"Unknown ensemble localizer members in {candidates!r}; "
                f"supported names are {supported!r}.",
            ) from exception
