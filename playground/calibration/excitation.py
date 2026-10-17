"""Reusable amplitude and electronics excitation programs for calibration studies."""

from __future__ import annotations

from math import isfinite

from numpy import eye, zeros

from .known_reference import ReferenceSegment


def multi_amplitude_reference_program(
        *,
        acceleration_magnitudes_mps2: tuple[float, ...] = (0.5, 2.0, 5.0, 9.80665, 15.0),
        angular_rate_magnitudes_radps: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 5.0),
        sample_count: int = 50,
        sample_period_s: float = 0.01,
        temperature_celsius: float | None = None,
) -> tuple[ReferenceSegment, ...]:
    """Return symmetric multi-axis amplitude sweeps for nonlinear/electronics effects."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if temperature_celsius is not None and not isfinite(temperature_celsius):
        raise ValueError("temperature_celsius must be finite when supplied")
    ####
    for values, field_name in (
        (acceleration_magnitudes_mps2, "acceleration_magnitudes_mps2"),
        (angular_rate_magnitudes_radps, "angular_rate_magnitudes_radps"),
    ):
        if not values or any((not isfinite(value) or value <= 0.0) for value in values):
            raise ValueError(f"{field_name} must contain finite positive magnitudes")
        ####
    ####
    axes = eye(3)
    duration_s = sample_count * sample_period_s
    segments: list[ReferenceSegment] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for magnitude in acceleration_magnitudes_mps2:
            for sign, sign_name in ((1.0, "p"), (-1.0, "a")):
                segments.append(
                    ReferenceSegment(
                        label=f"acc_{axis_name}_{sign_name}_{magnitude:g}",
                        duration_s=duration_s,
                        sample_count=sample_count,
                        specific_force_body=sign * magnitude * axes[axis_index],
                        angular_rate_body=zeros(3),
                        temperature_celsius=temperature_celsius,
                    )
                )
            ####
        ####
        for magnitude in angular_rate_magnitudes_radps:
            for sign, sign_name in ((1.0, "p"), (-1.0, "a")):
                segments.append(
                    ReferenceSegment(
                        label=f"gyr_{axis_name}_{sign_name}_{magnitude:g}",
                        duration_s=duration_s,
                        sample_count=sample_count,
                        specific_force_body=zeros(3),
                        angular_rate_body=sign * magnitude * axes[axis_index],
                        temperature_celsius=temperature_celsius,
                    )
                )
            ####
        ####
    ####
    return tuple(segments)
####
