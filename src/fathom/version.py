from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

from fathom import __version__


class VersionInfo:
    """
    Resolve installed package metadata for runtime logging.
    """

    @classmethod
    def payload(cls) -> Dict[str, Any]:
        """
        Return installed package metadata useful for deployment validation.
        """

        payload: Dict[str, Any] = {
            "name": "fathom",
            "source.version": __version__,
            "module.path": str(Path(__file__).resolve().parent),
        }

        try:
            distribution = metadata.distribution("fathom")
        except metadata.PackageNotFoundError:
            return payload

        payload["distribution.version"] = distribution.version

        if direct_url_text := distribution.read_text("direct_url.json"):
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError:
                payload["direct.url"] = direct_url_text
            else:
                payload["direct.url"] = direct_url.get("url")
                vcs_info = direct_url.get("vcs_info") or {}
                payload["git.commit"] = vcs_info.get("commit_id")
                payload["requested.revision"] = vcs_info.get("requested_revision")

        return payload


if __name__ == "__main__":
    print(json.dumps(VersionInfo.payload(), indent=2, sort_keys=True))
