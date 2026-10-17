"""Moving-block bootstrap covariance for serially correlated calibration records."""

from __future__ import annotations

from dataclasses import dataclass, replace

from numpy import arange, asarray, concatenate, cov, flatnonzero, isfinite, ndarray, random
from numpy.linalg import eigh

from .known_reference import (
    CalibrationDataset,
    CalibrationEstimate,
    calibration_coefficient_vectors,
    fit_known_reference_calibration,
)


def _ordered_unique_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(labels))
####


def _circular_block_resample(
        indices: ndarray,
        *,
        block_length: int,
        rng: random.Generator,
) -> ndarray:
    """Return one same-length circular moving-block resample of indices."""
    count = int(indices.size)
    block_count = (count + block_length - 1) // block_length
    starts = rng.integers(0, count, size=block_count)
    local_offsets = arange(block_length, dtype=int)
    local_indices = ((starts[:, None] + local_offsets[None, :]) % count).reshape(-1)[:count]
    return indices[local_indices]
####


def _resample_by_label(
        dataset: CalibrationDataset,
        *,
        block_length: int,
        rng: random.Generator,
) -> CalibrationDataset:
    labels_array = asarray(dataset.labels, dtype=object)
    groups = tuple(
        flatnonzero(labels_array == label)
        for label in _ordered_unique_labels(dataset.labels)
    )
    resampled_indices = concatenate(
        tuple(
            _circular_block_resample(indices, block_length=block_length, rng=rng)
            for indices in groups
        )
    )
    return CalibrationDataset(
        labels=tuple(dataset.labels[int(index)] for index in resampled_indices),
        start_times=dataset.start_times[resampled_indices],
        end_times=dataset.end_times[resampled_indices],
        temperatures_celsius=dataset.temperatures_celsius[resampled_indices],
        specific_force_body=dataset.specific_force_body[resampled_indices],
        angular_rate_body=dataset.angular_rate_body[resampled_indices],
        measured_acceleration=dataset.measured_acceleration[resampled_indices],
        measured_angular_rate=dataset.measured_angular_rate[resampled_indices],
    )
####


def _covariance_from_replicates(values: ndarray) -> ndarray:
    samples = asarray(values, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2 or not bool(isfinite(samples).all()):
        raise ValueError("bootstrap replicate values must be finite with shape (n, p), n >= 2")
    ####
    covariance = asarray(cov(samples, rowvar=False, ddof=1), dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = eigh(covariance)
    covariance = (eigenvectors * eigenvalues.clip(min=0.0)) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)
####


@dataclass(frozen=True, slots=True)
class CalibrationBootstrapResult:
    """Calibration estimate whose coefficient covariance comes from block resampling."""

    estimate: CalibrationEstimate
    replicate_count: int
    block_length: int
    label_count: int
    accelerometer_median_variance_inflation: float
    gyroscope_median_variance_inflation: float

    def __post_init__(self) -> None:
        if self.replicate_count < 2:
            raise ValueError("replicate_count must be at least two")
        ####
        if self.block_length < 1 or self.label_count < 1:
            raise ValueError("block_length and label_count must be positive")
        ####
        for field_name in (
                "accelerometer_median_variance_inflation",
                "gyroscope_median_variance_inflation",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            ####
        ####
    ####
####


def _median_variance_inflation(bootstrap_covariance: ndarray, classical_covariance: ndarray) -> float:
    from numpy import diag, median

    bootstrap_variance = diag(bootstrap_covariance)
    classical_variance = diag(classical_covariance)
    reference_scale = max(
        float(classical_variance.max(initial=0.0)),
        float(bootstrap_variance.max(initial=0.0)),
        1e-30,
    )
    usable = classical_variance > reference_scale * 1e-12
    if not bool(usable.any()):
        return 1.0
    ####
    return float(median(bootstrap_variance[usable] / classical_variance[usable]))
####


def moving_block_bootstrap_calibration(
        dataset: CalibrationDataset,
        *,
        replicate_count: int = 200,
        block_length: int = 10,
        seed: int = 0,
) -> CalibrationBootstrapResult:
    """Estimate calibration coefficient covariance by label-wise circular blocks.

    Each known-reference label is resampled independently so its design vector
    remains valid while short-range serial dependence is retained within the
    segment. This is a finite-record uncertainty model; it does not recreate
    cross-label latent-state continuity or correct deterministic model mismatch.
    """
    if not isinstance(dataset, CalibrationDataset):
        raise TypeError("dataset must be a CalibrationDataset")
    ####
    if isinstance(replicate_count, bool) or not isinstance(replicate_count, int) or replicate_count < 2:
        raise ValueError("replicate_count must be an integer >= 2")
    ####
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    ####
    label_count = len(_ordered_unique_labels(dataset.labels))
    base = fit_known_reference_calibration(dataset)
    joint_vectors = []
    for replicate_seed in random.SeedSequence(seed).spawn(replicate_count):
        replicate = fit_known_reference_calibration(
            _resample_by_label(
                dataset,
                block_length=block_length,
                rng=random.default_rng(replicate_seed),
            )
        )
        accel_vector, gyro_vector = calibration_coefficient_vectors(replicate)
        joint_vectors.append(concatenate((accel_vector, gyro_vector)))
    ####
    joint_covariance = _covariance_from_replicates(asarray(joint_vectors, dtype=float))
    accelerometer_covariance = joint_covariance[:12, :12]
    gyroscope_covariance = joint_covariance[12:, 12:]
    cross_covariance = joint_covariance[:12, 12:]
    estimate = replace(
        base,
        accelerometer_coefficient_covariance=accelerometer_covariance,
        gyroscope_coefficient_covariance=gyroscope_covariance,
        accelerometer_gyroscope_coefficient_cross_covariance=cross_covariance,
        covariance_assumption=(
            "segmentwise_circular_moving_block_bootstrap"
            f"(block_length={block_length},replicates={replicate_count})"
        ),
    )
    return CalibrationBootstrapResult(
        estimate=estimate,
        replicate_count=replicate_count,
        block_length=block_length,
        label_count=label_count,
        accelerometer_median_variance_inflation=_median_variance_inflation(
            accelerometer_covariance,
            base.accelerometer_coefficient_covariance,
        ),
        gyroscope_median_variance_inflation=_median_variance_inflation(
            gyroscope_covariance,
            base.gyroscope_coefficient_covariance,
        ),
    )
####
