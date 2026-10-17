"""Controlled motion contamination for calibration and alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sin

from numpy import asarray, isfinite as array_isfinite, ndarray


def _readonly_vector3(value: ndarray, field_name: str) -> ndarray:
    vector = asarray(value, dtype=float).copy()
    if vector.shape != (3,) or not bool(array_isfinite(vector).all()):
        raise ValueError(f"{field_name} must be a finite vector with shape (3,)")
    ####
    vector.setflags(write=False)
    return vector
####


@dataclass(frozen=True, slots=True)
class HarmonicMotionExcitation:
    """Sinusoidal specific-force and angular-rate contamination in body axes."""

    specific_force_amplitude_body: ndarray
    angular_rate_amplitude_body: ndarray
    frequency_hz: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        ####
        if not isfinite(self.phase_rad):
            raise ValueError("phase_rad must be finite")
        ####
        object.__setattr__(
            self,
            "specific_force_amplitude_body",
            _readonly_vector3(self.specific_force_amplitude_body, "specific_force_amplitude_body"),
        )
        object.__setattr__(
            self,
            "angular_rate_amplitude_body",
            _readonly_vector3(self.angular_rate_amplitude_body, "angular_rate_amplitude_body"),
        )
    ####

    def __call__(self, timestamp_s: float) -> tuple[ndarray, ndarray]:
        """Return the instantaneous contamination at one timestamp."""
        if not isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        ####
        value = sin(2.0 * pi * self.frequency_hz * timestamp_s + self.phase_rad)
        return (
            value * self.specific_force_amplitude_body,
            value * self.angular_rate_amplitude_body,
        )
    ####
####
