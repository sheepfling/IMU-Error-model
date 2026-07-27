from datetime import date
from typing import Annotated, Self

from numpy import all as all_values, allclose, asarray, isfinite, linalg
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationInfo, field_validator, model_validator


DEFAULT_REFERENCE_TEMPERATURE_C = 25.0

AxisValue = float | tuple[float, float, float]
Covariance3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class AxisConfig(BaseModel):
    """Noise and deterministic error parameters for one 3-axis channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    white_noise_density: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="White-noise amplitude spectral density in sensor units per square root second; must be non-negative.",
        ),
    ] = 0.0
    turn_on_bias_std: Annotated[
        AxisValue,
        Field(
            description="One-sigma turn-on bias in sensor units, as a scalar broadcast to all axes or a finite non-negative three-axis vector."
        ),
    ] = 0.0
    bias_std: Annotated[
        AxisValue,
        Field(
            description="One-sigma stationary Gauss-Markov bias in sensor units, or random-walk driving density in sensor units per square root second, as a scalar broadcast to all axes or a finite non-negative three-axis vector."
        ),
    ] = 0.0
    bias_correlation_time: Annotated[
        float | None,
        Field(
            gt=0.0,
            allow_inf_nan=False,
            description="Gauss-Markov bias correlation time in seconds; must be positive when provided.",
        ),
    ] = None
    flicker_bias_std: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="Target one-sigma flicker-bias level in sensor units; must be non-negative.",
        ),
    ] = 0.0
    flicker_min_correlation_time: Annotated[
        float | None,
        Field(
            gt=0.0,
            allow_inf_nan=False,
            description="Lower bound of the flicker correlation-time band in seconds; must be positive when provided.",
        ),
    ] = None
    flicker_max_correlation_time: Annotated[
        float | None,
        Field(
            gt=0.0,
            allow_inf_nan=False,
            description="Upper bound of the flicker correlation-time band in seconds; must be positive and greater than the lower bound when provided.",
        ),
    ] = None
    flicker_components: Annotated[
        int,
        Field(
            ge=2,
            description="Number of first-order components used to approximate flicker bias; must be at least 2 when flicker bias is enabled.",
        ),
    ] = 8
    scale_factor: Annotated[
        AxisValue,
        Field(
            description="Dimensionless multiplicative scale-factor error, as a scalar broadcast to all axes or a finite three-axis vector."
        ),
    ] = 0.0
    nonlinear_factor: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description="Dimensionless quadratic nonlinearity coefficient applied to the squared signal magnitude.",
        ),
    ] = 0.0
    misalignment_std: Annotated[
        float,
        Field(
            ge=0.0,
            allow_inf_nan=False,
            description="Isotropic one-sigma random rotation-vector misalignment angle in radians; use misalignment_covariance for anisotropic uncertainty.",
        ),
    ] = 0.0
    misalignment_covariance: Annotated[
        Covariance3 | None,
        Field(
            description="Optional symmetric positive-semidefinite 3x3 covariance of the reset rotation vector in radians squared; mutually exclusive with non-zero misalignment_std."
        ),
    ] = None
    measurement_range: Annotated[
        float | None,
        Field(
            gt=0.0,
            allow_inf_nan=False,
            description="Symmetric channel measurement limit in sensor units; clipping is disabled when omitted and the value must be positive when provided.",
        ),
    ] = None
    thermal_bias_coefficient: Annotated[
        float | tuple[float, float, float],
        Field(
            description="Bias temperature coefficient in sensor units per degree Celsius, as a scalar or finite three-axis vector."
        ),
    ] = 0.0
    thermal_noise_coefficient: Annotated[
        float | tuple[float, float, float],
        Field(
            description="Noise scale temperature coefficient per degree Celsius, as a scalar or finite three-axis vector."
        ),
    ] = 0.0
    thermal_scale_factor_coefficient: Annotated[
        float | tuple[float, float, float],
        Field(
            description="Scale-factor temperature coefficient per degree Celsius, as a scalar or finite three-axis vector."
        ),
    ] = 0.0
    reference_temperature_celsius: Annotated[
        float,
        Field(
            allow_inf_nan=False,
            description="Temperature reference point in degrees Celsius for thermal error coefficients.",
        ),
    ] = DEFAULT_REFERENCE_TEMPERATURE_C
    noise_covariance: Annotated[
        tuple[tuple[float, float, float], ...] | None,
        Field(
            description="Optional symmetric positive-semidefinite 3x3 covariance matrix for channel noise; omitted uses independent noise."
        ),
    ] = None
    quantization_step: Annotated[
        AxisValue | None,
        Field(
            description="Quantization step in sensor units, as a positive scalar broadcast to all axes or a finite positive three-axis vector; omitted disables quantization."
        ),
    ] = None
    apply_clipping: Annotated[
        bool,
        Field(
            description="Whether to clip the channel output to plus or minus measurement_range when a range is configured."
        ),
    ] = True
    output_scale: Annotated[
        float, Field(gt=0.0, description="Positive finite output scale applied to this channel's reported increments.")
    ] = 1.0

    @field_validator("turn_on_bias_std", "bias_std", "scale_factor", "quantization_step")
    @classmethod
    def finite_axis_value(cls, value: AxisValue | None, info: ValidationInfo) -> AxisValue | None:
        if value is None:
            return None
        ####
        values = asarray(value, dtype=float)
        if values.shape not in ((), (3,)) or not bool(all_values(isfinite(values))):
            raise ValueError(f"{info.field_name} must be a finite scalar or 3-vector")
        ####
        if info.field_name in {"turn_on_bias_std", "bias_std"} and not bool(all_values(values >= 0)):
            raise ValueError(f"{info.field_name} must be non-negative")
        ####
        if info.field_name == "quantization_step" and not bool(all_values(values > 0)):
            raise ValueError("quantization_step must be positive when provided")
        ####
        if values.ndim == 0:
            return float(values.item())
        ####
        return (float(values[0]), float(values[1]), float(values[2]))
    ####


    @field_validator("thermal_bias_coefficient", "thermal_noise_coefficient", "thermal_scale_factor_coefficient")
    @classmethod
    def finite_thermal_coefficient(cls, value: AxisValue) -> AxisValue:
        values = asarray(value, dtype=float)
        if values.shape not in ((), (3,)) or not bool(all_values(isfinite(values))):
            raise ValueError("thermal coefficients must be a finite scalar or 3-vector")
        ####
        return value
    ####


    @field_validator("output_scale")
    @classmethod
    def finite_output_scale(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("output_scale must be finite")
        ####
        return value
    ####


    @field_validator("misalignment_covariance")
    @classmethod
    def validate_misalignment_covariance(cls, value: Covariance3 | None) -> Covariance3 | None:
        if value is None:
            return None
        ####
        covariance = asarray(value, dtype=float)
        if (
            covariance.shape != (3, 3)
            or not bool(all_values(isfinite(covariance)))
            or not allclose(covariance, covariance.T)
        ):
            raise ValueError("misalignment_covariance must be a symmetric 3x3 matrix")
        ####
        if linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("misalignment_covariance must be positive semidefinite")
        ####
        return (
            (
                float(covariance[0, 0]),
                float(covariance[0, 1]),
                float(covariance[0, 2]),
            ),
            (
                float(covariance[1, 0]),
                float(covariance[1, 1]),
                float(covariance[1, 2]),
            ),
            (
                float(covariance[2, 0]),
                float(covariance[2, 1]),
                float(covariance[2, 2]),
            ),
        )
    ####


    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if self.flicker_bias_std > 0:
            if self.flicker_min_correlation_time is None or self.flicker_max_correlation_time is None:
                raise ValueError("flicker correlation-time bounds are required when flicker_bias_std is non-zero")
            ####
            if self.flicker_max_correlation_time <= self.flicker_min_correlation_time:
                raise ValueError("flicker correlation-time bounds must be positive and increasing")
            ####
            if self.flicker_components < 2:
                raise ValueError("flicker_components must be at least two")
            ####
        ####
        if self.noise_covariance is not None:
            covariance = asarray(self.noise_covariance, dtype=float)
            if (
                covariance.shape != (3, 3)
                or not bool(all_values(isfinite(covariance)))
                or not allclose(covariance, covariance.T)
            ):
                raise ValueError("noise_covariance must be a symmetric 3x3 matrix")
            ####
            if linalg.eigvalsh(covariance).min() < -1e-12:
                raise ValueError("noise_covariance must be positive semidefinite")
            ####
        ####
        if self.misalignment_covariance is not None and self.misalignment_std != 0.0:
            raise ValueError("misalignment_covariance cannot be combined with non-zero misalignment_std")
        ####
        return self
    ####
####


class ImuConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    accelerometer: Annotated[
        AxisConfig, Field(description="Configuration for the three-axis accelerometer channel.")
    ] = AxisConfig()
    gyroscope: Annotated[AxisConfig, Field(description="Configuration for the three-axis gyroscope channel.")] = (
        AxisConfig()
    )
    output_scale_accelerometer: Annotated[
        float,
        Field(
            gt=0.0, allow_inf_nan=False, description="Positive multiplier applied to reported accelerometer increments."
        ),
    ] = 1.0
    output_scale_gyroscope: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False, description="Positive multiplier applied to reported gyroscope increments."),
    ] = 1.0
####




class ProfileSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: Annotated[HttpUrl, Field(description="Canonical public URL for the referenced page or file.")]
    date: Annotated[date, Field(description="Date on which the source was reviewed for this profile.")]
    archive_urls: Annotated[
        tuple[HttpUrl, ...],
        Field(description="Optional archive.org, archive.is, or other backup URLs for this source."),
    ] = ()
####




class ProfileMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Annotated[str, Field(min_length=1, description="Normalized hardware or sensor family identifier.")]
    vendor: Annotated[str, Field(min_length=1, description="Manufacturer or effective system vendor name.")]
    grade: Annotated[str, Field(min_length=1, description="Intended performance or use classification.")]
    revision: Annotated[str, Field(min_length=1, description="Source, product, or profile revision identifier.")]
    source_date: Annotated[date, Field(description="Profile-level provenance review date.")]
    sources: Annotated[
        tuple[ProfileSource, ...],
        Field(min_length=1, description="One or more public source references supporting the profile estimates."),
    ]
    tags: Annotated[tuple[str, ...], Field(min_length=1, description="Searchable classification tags for the profile.")]
    active: Annotated[bool, Field(description="Whether this profile is an active project baseline.")]
####




class ProfileDocument(BaseModel):
    """Validated profile envelope around the canonical IMU configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, Field(ge=1, description="Positive profile document schema version.")] = 1
    model_name: Annotated[str, Field(min_length=1, description="Human-readable profile model name.")]
    sample_period_s: Annotated[
        float,
        Field(gt=0.0, allow_inf_nan=False, description="Nominal sample period in seconds."),
    ]
    metadata: Annotated[ProfileMetadata, Field(description="Validated provenance and classification metadata.")]
    accelerometer: Annotated[AxisConfig, Field(description="Canonical accelerometer configuration.")] = Field(
        default_factory=AxisConfig
    )
    gyroscope: Annotated[AxisConfig, Field(description="Canonical gyroscope configuration.")] = Field(
        default_factory=AxisConfig
    )
####
