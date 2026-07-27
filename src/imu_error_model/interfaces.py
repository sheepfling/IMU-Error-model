"""Public extension points for alternate IMU measurement models."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from numpy import ndarray

from .checkpoint_codecs import CheckpointCodecProtocol
from .signals import ImuOutput


CheckpointT = TypeVar("CheckpointT")

@runtime_checkable
class ImuModelProtocol(Protocol):
    """Structural interface for truth-state-driven IMU measurement models.

    Implementations consume gravity-excluded truth velocity and orientation,
    then return body-frame increments over the interval since the previous
    call. Runtime checks only verify that `reset` and `measure` attributes
    exist; static type checking validates their signatures.
    """

    def reset(self) -> None:
        """Reset latent stochastic processes and the interval baseline."""
        ...

    def measure(
            self,
            timestamp: float,
            velocity_without_gravity: ndarray,
            orientation_world_from_body: ndarray,
            temperature_celsius: float | None = None,
    ) -> ImuOutput:
        """Return noisy body-frame increments for the current truth state."""
        ...
####


@runtime_checkable
class CheckpointableImuModelProtocol(Protocol[CheckpointT]):
    """Structural interface for models that can pause and resume exactly."""

    def snapshot(self) -> CheckpointT:
        """Return a complete, validated model-state checkpoint."""
        ...

    def restore(self, checkpoint: CheckpointT) -> None:
        """Restore a previously captured checkpoint into this model."""
        ...
####


@runtime_checkable
class SerializableCheckpointableImuModelProtocol(
        CheckpointableImuModelProtocol[CheckpointT], Protocol[CheckpointT]
):
    """Checkpoint protocol with an explicitly discoverable byte codec."""

    checkpoint_codec: CheckpointCodecProtocol[CheckpointT]
####
