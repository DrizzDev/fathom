from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

from google.cloud import storage

from fathom.exceptions import VisionError
from fathom.interfaces import IImageStorage
from fathom.schemas.configuration import GeminiConfig

logger = getLogger(__name__)


class GCSImageStorage(IImageStorage):
    """
    Handles uploading images to Google Cloud Storage.
    """

    def __init__(self, configuration: GeminiConfig, credentials: Any) -> None:
        self.__credentials = credentials
        self.__configuration = configuration

    async def save(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Uploads image to GCS and returns the URI.
        Path: YYYY-MM-DD/{package}/{session}/{timestamp}__{activity}.png
        """

        credentials = self.__credentials
        bucket = self.__configuration.gcs_bucket
        project = self.__configuration.project_id

        def __upload_sync() -> str:
            """
            Uploads image to GCS and returns the URI.
            """

            try:
                client = storage.Client(project=project, credentials=credentials)
                storage_bucket = client.bucket(bucket)

                package = "unknown_app"
                session = "default_session"
                activity = "unknown_screen"
                folder = datetime.now().strftime("%Y-%m-%d")

                if metadata:
                    session = metadata.get("session_id") or session
                    package = metadata.get("package_name") or package
                    activity = metadata.get("activity_name") or activity

                # Sanitize
                package = "".join(char for char in package if char.isalnum() or char in "._-")
                session = "".join(char for char in session if char.isalnum() or char in "._-")
                activity = "".join(char for char in activity if char.isalnum() or char in "._-")

                timestamp = int(time.time() * 1000)
                filename = f"{folder}/{package}/{session}/{timestamp}__{activity}.png"

                blob = storage_bucket.blob(filename)
                blob.upload_from_string(data, content_type="image/png")

                uri = f"gs://{bucket}/{filename}"
                logger.debug(f"Uploaded image to GCS: {uri}")
                return uri
            except Exception as exception:
                logger.warning(f"Failed to upload to GCS: {exception}")
                raise VisionError(f"GCS upload failed: {exception}") from exception

        return await asyncio.to_thread(__upload_sync)
