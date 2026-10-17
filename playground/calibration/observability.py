"""Calibration design observability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from numpy import column_stack, ndarray, ones
from numpy.linalg import matrix_rank, svd

from .known_reference import CalibrationDataset


@dataclass(frozen=True, slots=True)
class DesignObservability:
    """Rank, conditioning, and weak-direction diagnostics for a calibration design."""

    rank: int
    parameter_count: int
    sample_count: int
    singular_values: ndarray
    condition_number: float
    weakest_right_singular_vector: ndarray

    @property
    def full_rank(self) -> bool:
        return self.rank == self.parameter_count
    ####
####


def _analyze_design(design: ndarray) -> DesignObservability:
    u, singular_values, vh = svd(design, full_matrices=False)
    del u
    rank = int(matrix_rank(design))
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = float("inf") if smallest <= 0.0 else float(singular_values[0] / smallest)
    singular_values = singular_values.copy()
    weakest = vh[-1].copy()
    singular_values.setflags(write=False)
    weakest.setflags(write=False)
    return DesignObservability(
        rank=rank,
        parameter_count=design.shape[1],
        sample_count=design.shape[0],
        singular_values=singular_values,
        condition_number=condition_number,
        weakest_right_singular_vector=weakest,
    )
####


def analyze_known_reference_observability(
        dataset: CalibrationDataset,
) -> tuple[DesignObservability, DesignObservability]:
    """Return accelerometer and gyro design observability for a corpus."""
    sample_count = len(dataset.labels)
    accel_design = column_stack((dataset.specific_force_body, ones(sample_count)))
    gyro_design = column_stack((dataset.angular_rate_body, dataset.specific_force_body, ones(sample_count)))
    return _analyze_design(accel_design), _analyze_design(gyro_design)
####
