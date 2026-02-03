from __future__ import annotations

import asyncio
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any

from google.cloud import storage

from fathom.exceptions import VisionError
from fathom.interfaces import IImageStorage
from fathom.schemas.configuration import GeminiConfig

logger = getLogger(__name__)


class LocalImageStorage(IImageStorage):
    """
    Handles saving images to the local filesystem.
    """

    async def save(self, data: bytes) -> str:
        """
        Saves image data locally with a human-readable timestamp.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.__build_path(timestamp)
            self.__write(path, data)
            logger.debug(f"Saved local screenshot: {path}")
            return str(path.absolute())
        except Exception as exception:
            logger.warning(f"Failed to save local screenshot: {exception}")
            raise VisionError(f"Local save failed: {exception}") from exception

    def __build_path(self, timestamp: str) -> Path:
        """
        Constructs the destination path for the screenshot.
        """
        directory = Path("assets/screenshot")
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{timestamp}.png"

    def __write(self, path: Path, data: bytes) -> None:
        """
        Handles physical write to disk.
        """
        with path.open("wb") as handle:
            handle.write(data)


class GCSImageStorage(IImageStorage):
    """
    Handles uploading images to Google Cloud Storage.
    """

    def __init__(self, configuration: GeminiConfig, credentials: Any) -> None:
        self.__configuration = configuration
        self.__credentials = credentials

    async def save(self, data: bytes) -> str:
        """
        Uploads image to GCS and returns the URI.
        """
        project = self.__configuration.project_id
        bucket = self.__configuration.gcs_bucket
        credentials = self.__credentials

        def __upload_sync() -> str:
            try:
                client = storage.Client(project=project, credentials=credentials)
                storage_bucket = client.bucket(bucket)
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}.png"
                blob = storage_bucket.blob(filename)
                blob.upload_from_string(data, content_type="image/png")
                uri = f"gs://{bucket}/{filename}"
                logger.debug(f"Uploaded image to GCS: {uri}")
                return uri
            except Exception as exception:
                logger.warning(f"Failed to upload to GCS: {exception}")
                raise VisionError(f"GCS upload failed: {exception}") from exception

        return await asyncio.to_thread(__upload_sync)
