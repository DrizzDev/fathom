from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

from google.cloud import storage

from fathom.core.exceptions import VisionError
from fathom.infrastructure.storage.metadata import extract_metadata
from fathom.interfaces import IImageStorage
from fathom.schemas.configuration import StorageConfiguration

logger = getLogger(__name__)


class GCSImageStorage(IImageStorage):
    """
    Handles uploading images to Google Cloud Storage.
    """

    def __init__(self, configuration: StorageConfiguration) -> None:
        """
        Initialize GCS storage with configuration and credentials.
        """

        self.__configuration = configuration

    async def save(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload bytes to GCS using path layout {category}/{date}/{session}/{filename}.
        """

        project = self.__configuration.project_id
        bucket = self.__configuration.storage_bucket
        credentials = self.__configuration.credentials

        def __upload_sync() -> str:
            """
            Synchronously perform the upload on a worker thread.
            """

            try:
                client_kwargs = {}
                if project:
                    client_kwargs["project"] = project

                if isinstance(credentials, str):
                    client = storage.Client.from_service_account_json(credentials, **client_kwargs)

                elif isinstance(credentials, dict):
                    client = storage.Client.from_service_account_info(credentials, **client_kwargs)

                else:
                    # Use positional argument for project if available
                    client = storage.Client(project=project) if project else storage.Client()

                meta = extract_metadata(metadata or {})

                storage_bucket = client.bucket(bucket)

                folder = datetime.now().strftime("%Y-%m-%d")
                computed_name = meta.filename or self.__fallback_filename(
                    category=meta.category, activity=meta.activity, package=meta.package
                )
                filename = f"{meta.category}/{folder}/{meta.session}/{computed_name}"

                content_type = self.__content_type_for(filename=filename)

                blob = storage_bucket.blob(filename)
                blob.upload_from_string(data, content_type=content_type)

                uri = f"gs://{bucket}/{filename}"
                logger.debug(f"Uploaded image to GCS: {uri}")
                return uri
            except Exception as exception:
                logger.warning(f"Failed to upload to GCS: {exception}")
                raise VisionError(f"GCS upload failed: {exception}") from exception

        return await asyncio.to_thread(__upload_sync)

    @staticmethod
    def __fallback_filename(*, category: str, activity: str, package: str) -> str:
        """
        Build a category-aware filename when the caller did not supply one.
        Package is embedded so multi-package sessions do not collide.
        """

        extension = GCSImageStorage.__extension_for(category=category)

        timestamp = int(time.time() * 1000)
        return f"{timestamp}__{package}__{activity}{extension}"

    @staticmethod
    def __extension_for(*, category: str) -> str:
        """
        Resolve the file extension for an artifact category.
        """

        if category == "xmls":
            return ".xml"

        if category == "history":
            return ".txt"

        return ".png"

    @staticmethod
    def __content_type_for(*, filename: str) -> str:
        """
        Resolve the HTTP content-type for an uploaded GCS object from its extension.
        """

        if filename.endswith(".png"):
            return "image/png"

        if filename.endswith(".json"):
            return "application/json"

        if filename.endswith(".yaml") or filename.endswith(".yml"):
            return "text/yaml"

        if filename.endswith(".txt"):
            return "text/plain"

        if filename.endswith(".xml"):
            return "application/xml"

        return "application/octet-stream"
