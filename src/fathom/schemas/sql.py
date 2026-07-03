from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TypeAlias, Union

from pydantic import JsonValue

SqlParameterValue: TypeAlias = Union[JsonValue, date, datetime, Decimal, bytes]
