from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RuntimeCheckpoint(BaseModel):
    """
    Serialized runtime state used for graph checkpoint persistence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1, description="Checkpoint schema version.")
    payload: JsonValue = Field(description="Versioned runtime checkpoint payload.")


class CheckpointCodec:
    """
    Encodes and decodes runtime state checkpoints.
    """

    def encode(self, *, payload: JsonValue, version: int = 1) -> RuntimeCheckpoint:
        """
        Encode a validated checkpoint payload.
        """

        return RuntimeCheckpoint(payload=payload, version=version)

    def decode(self, *, checkpoint: RuntimeCheckpoint) -> JsonValue:
        """
        Decode a runtime checkpoint payload.
        """

        return checkpoint.payload
