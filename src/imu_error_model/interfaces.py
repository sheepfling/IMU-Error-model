"""Public extension points for alternate IMU measurement models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .signals import ImuOutput


class ImuModelProtocol(Protocol):
    """Structural interface for truth-state-driven IMU measurement models.

    Implementations consume truth velocity with gravity excluded and truth
    orientation, then return body-frame increments over the interval since
    the previous call.
    """

    def reset(self) -> None: ...

    def measure(
        self,
        timestamp: float,
        velocity_without_gravity: np.ndarray,
        orientation_world_from_body: np.ndarray,
        temperature: float = 25.0,
    ) -> ImuOutput: ...


class BaseImuModel(ABC):
    """Optional base class for models that want validation and shared lifecycle."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state, including latent stochastic processes."""

    @abstractmethod
    def measure(
        self,
        timestamp: float,
        velocity_without_gravity: np.ndarray,
        orientation_world_from_body: np.ndarray,
        temperature: float = 25.0,
    ) -> ImuOutput:
        """Return noisy body-frame increments for the current truth state."""
