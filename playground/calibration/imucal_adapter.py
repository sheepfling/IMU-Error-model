"""Optional adapters between synthetic playground data and :mod:`imucal`."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, pi, radians
from pathlib import Path
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from numpy import asarray, column_stack, degrees, diag, eye, ndarray, vstack, zeros
from numpy.linalg import inv

from imu_error_model import ImuConfig

from .known_reference import (
    CalibrationDataset,
    ReferenceSegment,
    _readonly_matrix,
    _readonly_vector3,
    effective_sensitivity_matrix,
    expected_intrinsic_vector,
)


if TYPE_CHECKING:
    import pandas as pd
####


@dataclass(frozen=True, slots=True)
class ImucalForwardParameters:
    """Canonical forward parameters reconstructed from an imucal result."""

    accelerometer_sensitivity_matrix: ndarray
    accelerometer_bias: ndarray
    gyroscope_sensitivity_matrix: ndarray
    gyroscope_specific_force_sensitivity: ndarray
    gyroscope_bias_radps: ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accelerometer_sensitivity_matrix",
            _readonly_matrix(
                self.accelerometer_sensitivity_matrix,
                (3, 3),
                "accelerometer_sensitivity_matrix",
            ),
        )
        object.__setattr__(
            self,
            "accelerometer_bias",
            _readonly_vector3(self.accelerometer_bias, "accelerometer_bias"),
        )
        object.__setattr__(
            self,
            "gyroscope_sensitivity_matrix",
            _readonly_matrix(self.gyroscope_sensitivity_matrix, (3, 3), "gyroscope_sensitivity_matrix"),
        )
        object.__setattr__(
            self,
            "gyroscope_specific_force_sensitivity",
            _readonly_matrix(
                self.gyroscope_specific_force_sensitivity,
                (3, 3),
                "gyroscope_specific_force_sensitivity",
            ),
        )
        object.__setattr__(
            self,
            "gyroscope_bias_radps",
            _readonly_vector3(self.gyroscope_bias_radps, "gyroscope_bias_radps"),
        )
    ####
####


def ferraris_imucal_program(
        *,
        sampling_rate_hz: float = 100.0,
        static_duration_s: float = 2.0,
        nominal_rotation_rate_radps: float = pi / 2.0,
        expected_angle_degrees: float = -360.0,
        gravity_mps2: float = 9.81,
        temperature_celsius: float | None = None,
) -> tuple[ReferenceSegment, ...]:
    """Generate the nine regions expected by ``imucal.FerrarisCalibration``."""
    for field_name, value in (
        ("sampling_rate_hz", sampling_rate_hz),
        ("static_duration_s", static_duration_s),
        ("nominal_rotation_rate_radps", nominal_rotation_rate_radps),
        ("gravity_mps2", gravity_mps2),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be finite and positive")
        ####
    ####
    if not isfinite(expected_angle_degrees) or expected_angle_degrees == 0.0:
        raise ValueError("expected_angle_degrees must be finite and non-zero")
    ####
    static_sample_count = max(1, round(static_duration_s * sampling_rate_hz))
    static_duration = static_sample_count / sampling_rate_hz
    expected_angle_rad = radians(expected_angle_degrees)
    rotation_sample_count = max(
        1,
        ceil(abs(expected_angle_rad) / nominal_rotation_rate_radps * sampling_rate_hz),
    )
    rotation_duration = rotation_sample_count / sampling_rate_hz
    angular_rate_radps = expected_angle_rad / rotation_duration
    axes = eye(3)
    segments: list[ReferenceSegment] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        segments.extend(
            (
                ReferenceSegment(
                    label=f"{axis_name}_p",
                    duration_s=static_duration,
                    sample_count=static_sample_count,
                    specific_force_body=gravity_mps2 * axes[axis_index],
                    angular_rate_body=zeros(3),
                    temperature_celsius=temperature_celsius,
                ),
                ReferenceSegment(
                    label=f"{axis_name}_a",
                    duration_s=static_duration,
                    sample_count=static_sample_count,
                    specific_force_body=-gravity_mps2 * axes[axis_index],
                    angular_rate_body=zeros(3),
                    temperature_celsius=temperature_celsius,
                ),
            )
        )
    ####
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        segments.append(
            ReferenceSegment(
                label=f"{axis_name}_rot",
                duration_s=rotation_duration,
                sample_count=rotation_sample_count,
                specific_force_body=gravity_mps2 * axes[axis_index],
                angular_rate_body=angular_rate_radps * axes[axis_index],
                temperature_celsius=temperature_celsius,
            )
        )
    ####
    return tuple(segments)
####


def to_imucal_dataframe(dataset: CalibrationDataset) -> pd.DataFrame:
    """Return imucal's six-column, region-indexed pandas input frame."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required for imucal dataframe export") from error
    ####
    gyro_degrees = degrees(dataset.measured_angular_rate)
    return pd.DataFrame(
        {
            "acc_x": dataset.measured_acceleration[:, 0],
            "acc_y": dataset.measured_acceleration[:, 1],
            "acc_z": dataset.measured_acceleration[:, 2],
            "gyr_x": gyro_degrees[:, 0],
            "gyr_y": gyro_degrees[:, 1],
            "gyr_z": gyro_degrees[:, 2],
        },
        index=pd.Index(dataset.labels, name="part"),
    )
####


def write_imucal_csv(dataset: CalibrationDataset, path: str | Path) -> None:
    """Write the exact region-indexed six-column CSV consumed by imucal."""
    to_imucal_dataframe(dataset).to_csv(Path(path))
####


def fit_ferraris_equation_reference(
        dataset: CalibrationDataset,
        *,
        sampling_rate_hz: float,
        gravity_mps2: float = 9.81,
        expected_angle_degrees: float = -360.0,
) -> ImucalForwardParameters:
    """Evaluate the Ferraris equations used by imucal without importing it.

    This small offline reference validates the corpus and unit/frame adapter. It
    is not a substitute for the dependency-gated `compute_imucal_ferraris`
    integration test.
    """
    if not isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    ####
    if not isfinite(gravity_mps2) or gravity_mps2 <= 0.0:
        raise ValueError("gravity_mps2 must be finite and positive")
    ####
    if not isfinite(expected_angle_degrees) or expected_angle_degrees == 0.0:
        raise ValueError("expected_angle_degrees must be finite and non-zero")
    ####

    required_labels = (
        "x_p",
        "x_a",
        "y_p",
        "y_a",
        "z_p",
        "z_a",
        "x_rot",
        "y_rot",
        "z_rot",
    )
    region_indices = {
        label: tuple(index for index, sample_label in enumerate(dataset.labels) if sample_label == label)
        for label in required_labels
    }
    missing = tuple(label for label, indices in region_indices.items() if not indices)
    if missing:
        raise ValueError(f"Ferraris dataset is missing regions: {', '.join(missing)}")
    ####

    def acceleration(label: str) -> ndarray:
        return dataset.measured_acceleration[region_indices[label], :]
    ####

    measured_gyro_degrees = degrees(dataset.measured_angular_rate)

    def gyroscope(label: str) -> ndarray:
        return measured_gyro_degrees[region_indices[label], :]
    ####

    accel_positive = column_stack(
        tuple(acceleration(f"{axis}_p").mean(axis=0) for axis in ("x", "y", "z"))
    )
    accel_antiparallel = column_stack(
        tuple(acceleration(f"{axis}_a").mean(axis=0) for axis in ("x", "y", "z"))
    )
    accelerometer_bias = diag((accel_positive + accel_antiparallel) / 2.0)
    accelerometer_matrix = (accel_positive - accel_antiparallel) / (2.0 * gravity_mps2)

    def calibrated_acceleration(label: str) -> ndarray:
        raw = acceleration(label)
        return (raw - accelerometer_bias) @ inv(accelerometer_matrix).T
    ####

    gyro_bias_degrees = vstack(
        tuple(gyroscope(label) for label in ("x_p", "x_a", "y_p", "y_a", "z_p", "z_a"))
    ).mean(axis=0)
    gyro_positive = column_stack(
        tuple(gyroscope(f"{axis}_p").mean(axis=0) for axis in ("x", "y", "z"))
    )
    gyro_antiparallel = column_stack(
        tuple(gyroscope(f"{axis}_a").mean(axis=0) for axis in ("x", "y", "z"))
    )
    gyro_force_sensitivity_degrees = (
        gyro_positive - gyro_antiparallel
    ) / (2.0 * gravity_mps2)

    integrated_gyro = zeros((3, 3))
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        label = f"{axis_name}_rot"
        calibrated_accel = calibrated_acceleration(label)
        offsets = (
            gyro_force_sensitivity_degrees @ calibrated_accel.T
        ).T + gyro_bias_degrees
        corrected_gyro = gyroscope(label) - offsets
        integrated_gyro[:, axis_index] = corrected_gyro.sum(axis=0) / sampling_rate_hz
    ####
    gyroscope_matrix = integrated_gyro / expected_angle_degrees
    degrees_to_radians = pi / 180.0
    return ImucalForwardParameters(
        accelerometer_sensitivity_matrix=accelerometer_matrix,
        accelerometer_bias=accelerometer_bias,
        gyroscope_sensitivity_matrix=gyroscope_matrix,
        gyroscope_specific_force_sensitivity=(
            gyro_force_sensitivity_degrees * degrees_to_radians
        ),
        gyroscope_bias_radps=gyro_bias_degrees * degrees_to_radians,
    )
####


def compute_imucal_ferraris(
        dataset: CalibrationDataset,
        *,
        sampling_rate_hz: float,
        gravity_mps2: float = 9.81,
        expected_angle_degrees: float = -360.0,
) -> Any:
    """Run imucal when the optional calibration dependency is installed."""
    try:
        imucal = cast(Any, import_module("imucal"))
        FerrarisCalibration = imucal.FerrarisCalibration
        ferraris_regions_from_df = imucal.ferraris_regions_from_df
    except ImportError as error:
        raise RuntimeError(
            "imucal is required for this comparison; install imu-error-model[calibration]"
        ) from error
    ####
    frame = to_imucal_dataframe(dataset)
    regions = ferraris_regions_from_df(frame)
    return FerrarisCalibration(
        grav=gravity_mps2,
        expected_angle=expected_angle_degrees,
    ).compute(
        regions,
        sampling_rate_hz=sampling_rate_hz,
        from_acc_unit="m/s^2",
        from_gyr_unit="deg/s",
        comment="Synthetic calibration generated by imu-error-model",
    )
####


def imucal_to_forward_parameters(calibration_info: Any) -> ImucalForwardParameters:
    """Convert imucal's ``K``/``R`` correction factors to the forward model."""
    required = ("K_a", "R_a", "b_a", "K_g", "R_g", "K_ga", "b_g")
    missing = tuple(name for name in required if getattr(calibration_info, name, None) is None)
    if missing:
        raise ValueError(f"imucal calibration is missing parameters: {', '.join(missing)}")
    ####
    degrees_to_radians = pi / 180.0
    return ImucalForwardParameters(
        accelerometer_sensitivity_matrix=(
            asarray(calibration_info.K_a, dtype=float) @ asarray(calibration_info.R_a, dtype=float)
        ),
        accelerometer_bias=asarray(calibration_info.b_a, dtype=float),
        gyroscope_sensitivity_matrix=(
            asarray(calibration_info.K_g, dtype=float) @ asarray(calibration_info.R_g, dtype=float)
        ),
        gyroscope_specific_force_sensitivity=(
            asarray(calibration_info.K_ga, dtype=float) * degrees_to_radians
        ),
        gyroscope_bias_radps=asarray(calibration_info.b_g, dtype=float) * degrees_to_radians,
    )
####


def imucal_parameter_error(parameters: ImucalForwardParameters, config: ImuConfig) -> ndarray:
    """Flatten canonical imucal-minus-injected deterministic parameter errors."""
    expected_gyro_force_sensitivity = (
        zeros((3, 3))
        if config.gyroscope_specific_force_sensitivity is None
        else asarray(config.gyroscope_specific_force_sensitivity, dtype=float)
    )
    return asarray(
        [
            *(parameters.accelerometer_sensitivity_matrix - effective_sensitivity_matrix(config.accelerometer)).ravel(),
            *(parameters.accelerometer_bias - expected_intrinsic_vector(config.accelerometer.fixed_bias)).ravel(),
            *(parameters.gyroscope_sensitivity_matrix - effective_sensitivity_matrix(config.gyroscope)).ravel(),
            *(parameters.gyroscope_specific_force_sensitivity - expected_gyro_force_sensitivity).ravel(),
            *(parameters.gyroscope_bias_radps - expected_intrinsic_vector(config.gyroscope.fixed_bias)).ravel(),
        ],
        dtype=float,
    )
####
