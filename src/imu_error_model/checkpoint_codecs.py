"""Format-specific codecs for transporting model checkpoints."""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


CheckpointT = TypeVar("CheckpointT")
PydanticCheckpointT = TypeVar("PydanticCheckpointT", bound=BaseModel)


@runtime_checkable
class CheckpointCodecProtocol(Protocol[CheckpointT]):
    """Encode and decode a model-defined checkpoint at a byte boundary."""

    media_type: str

    def encode(self, checkpoint: CheckpointT) -> bytes:
        """Serialize a validated checkpoint into transportable bytes."""
        ...

    def decode(self, payload: bytes) -> CheckpointT:
        """Deserialize and validate transport bytes into a checkpoint."""
        ...
####


class PydanticJsonCheckpointCodec(Generic[PydanticCheckpointT]):
    """Serialize a Pydantic checkpoint as UTF-8 JSON without executable loading."""

    media_type = "application/json"

    def __init__(self, checkpoint_type: type[PydanticCheckpointT]) -> None:
        self._checkpoint_type = checkpoint_type
    ####

    def encode(self, checkpoint: PydanticCheckpointT) -> bytes:
        """Return an indented UTF-8 JSON representation of a checkpoint."""
        return (checkpoint.model_dump_json(indent=2) + "\n").encode("utf-8")
    ####

    def decode(self, payload: bytes) -> PydanticCheckpointT:
        """Validate UTF-8 JSON bytes against the configured Pydantic model."""
        return self._checkpoint_type.model_validate_json(payload)
    ####
####
