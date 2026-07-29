from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ParsedModule(BaseModel):
    """
    A parsed source module: its filesystem path, repository-relative path, dotted package,
    and abstract syntax tree.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path = Field(description="Absolute filesystem path of the module.")
    relative: str = Field(description="Repository-relative path, used in reports.")
    package: str = Field(description="Dotted package name of the module.")
    tree: ast.Module = Field(description="Parsed abstract syntax tree of the module.")
