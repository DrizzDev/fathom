import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from fathom.exceptions import FathomError
from fathom.settings.env import FathomSettings


class GoogleCredentials(BaseModel):
    """
    Google Cloud Credentials structure.
    """

    type: str = Field(description="Credential type")
    project_id: str = Field(description="GCP Project ID")
    private_key_id: str = Field(description="Private key ID")
    private_key: str = Field(description="Private key content")
    client_email: str = Field(description="Service account email")

    auth_uri: str = Field(description="Auth URI")
    token_uri: str = Field(description="Token URI")
    client_id: str = Field(description="Client ID")

    client_x509_cert_url: str = Field(description="Client cert URL")
    auth_provider_x509_cert_url: str = Field(description="Auth provider cert URL")
    universe_domain: Optional[str] = Field(default="googleapis.com", description="Universe domain")


class CredentialsManager:
    """
    Manages loading and validation of credentials.
    """

    @staticmethod
    def load_google_credentials(file_path: Optional[str] = None) -> Optional[GoogleCredentials]:
        """
        Load Google credentials from file or environment.

        Args:
            file_path: Explicit path to credentials JSON.

        Returns:
            GoogleCredentials object or None if not found/invalid.

        Raises:
            FathomError: If file exists but is invalid.
        """

        target_path = file_path

        if not target_path:
            settings = FathomSettings()
            target_path = settings.google_application_credentials

        if not target_path:
            cwd_file = Path("credentials.json")
            if cwd_file.exists():
                target_path = str(cwd_file)
            else:
                repo_root = Path(__file__).parent.parent.parent.parent / "credentials.json"
                if repo_root.exists():
                    target_path = str(repo_root)

        if not target_path or not Path(target_path).exists():
            return None

        try:
            with Path(target_path).open(encoding="utf-8") as credentials_file:
                data = json.load(credentials_file)
                return GoogleCredentials(**data)
        except (json.JSONDecodeError, ValidationError) as exception:
            raise FathomError(f"Invalid credentials file at {target_path}") from exception

        except Exception as exception:
            raise FathomError(f"Failed to load credentials from {target_path}") from exception
