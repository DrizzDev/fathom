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
        Uploads image to GCS and returns the URI.
        Path: YYYY-MM-DD/{package}/{session}/{timestamp}__{activity}.png
        """

        project = self.__configuration.project_id
        bucket = self.__configuration.storage_bucket
        credentials = self.__configuration.credentials

        def __upload_sync() -> str:
            """
            Uploads image to GCS and returns the URI.
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

                if meta.filename:
<<<<<<< HEAD
                    filename = f"{meta.category}/{folder}/{meta.package}/{meta.session}/{meta.filename}"
                else:
                    timestamp = int(time.time() * 1000)
                    filename = (
                        f"{meta.category}/{folder}/{meta.package}/{meta.session}/{timestamp}__{meta.activity}.png"
                    )
=======
                    filename = (
                        f"{meta.category}/{folder}/{meta.package}/{meta.session}/{meta.filename}"
                    )
                else:
                    timestamp = int(time.time() * 1000)
                    filename = f"{meta.category}/{folder}/{meta.package}/{meta.session}/{timestamp}__{meta.activity}.png"
>>>>>>> 885ff26 (Deduplicate shared logic and fix design violations across the codebase)

                content_type = "application/octet-stream"

                if filename.endswith(".png"):
                    content_type = "image/png"

                elif filename.endswith(".json"):
                    content_type = "application/json"

                elif filename.endswith(".yaml") or filename.endswith(".yml"):
                    content_type = "text/yaml"

                elif filename.endswith(".txt"):
                    content_type = "text/plain"

                elif filename.endswith(".xml"):
                    content_type = "application/xml"

                blob = storage_bucket.blob(filename)
                blob.upload_from_string(data, content_type=content_type)

                uri = f"gs://{bucket}/{filename}"
                logger.debug(f"Uploaded image to GCS: {uri}")
                return uri
            except Exception as exception:
                logger.warning(f"Failed to upload to GCS: {exception}")
                raise VisionError(f"GCS upload failed: {exception}") from exception

        return await asyncio.to_thread(__upload_sync)
