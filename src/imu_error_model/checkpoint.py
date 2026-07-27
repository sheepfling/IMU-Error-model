"""Versioned checkpoints for resumable IMU model simulations."""

from math import isfinite
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import ImuConfig


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
FlickerStates = tuple[Vector3, ...]
RngBitGeneratorName = Literal["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

class ImuModelCheckpoint(BaseModel):
    """Complete resumable state for `imu_error_model.ImuModel`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, Field(ge=1, description="Checkpoint envelope schema version.")] = 1
    model_type: Annotated[
        Literal["imu_error_model.ImuModel"],
        Field(description="Model implementation identifier required to restore this checkpoint."),
    ] = "imu_error_model.ImuModel"
    config: Annotated[ImuConfig, Field(description="Canonical IMU configuration used by the model.")]
    rng_bit_generator: Annotated[
        RngBitGeneratorName,
        Field(description="Supported NumPy bit-generator class name used by the model RNG."),
    ]
    rng_state: Annotated[
        dict[str, JsonValue],
        Field(description="JSON-compatible NumPy bit-generator state using only nested JSON values."),
    ]
    accelerometer_turn_on_bias: Annotated[Vector3, Field(description="Sampled accelerometer turn-on bias state.")]
    gyroscope_turn_on_bias: Annotated[Vector3, Field(description="Sampled gyroscope turn-on bias state.")]
    accelerometer_bias: Annotated[Vector3, Field(description="Current accelerometer in-run bias state.")]
    gyroscope_bias: Annotated[Vector3, Field(description="Current gyroscope in-run bias state.")]
    accelerometer_flicker_states: Annotated[
        FlickerStates,
        Field(description="Current accelerometer finite-band flicker component states."),
    ]
    gyroscope_flicker_states: Annotated[
        FlickerStates,
        Field(description="Current gyroscope finite-band flicker component states."),
    ]
    accelerometer_misalignment: Annotated[Matrix3, Field(description="Sampled accelerometer misalignment matrix.")]
    gyroscope_misalignment: Annotated[Matrix3, Field(description="Sampled gyroscope misalignment matrix.")]
    previous_timestamp: Annotated[float | None, Field(description="Timestamp of the previous truth sample.")]
    previous_velocity: Annotated[Vector3 | None, Field(description="Previous gravity-excluded truth velocity.")]
    previous_orientation: Annotated[Matrix3 | None, Field(description="Previous world-from-body orientation matrix.")]
    thermal_model_type: Annotated[
        Literal["linear"],
        Field(description="Serializable thermal-model implementation identifier."),
    ] = "linear"

    @field_validator("previous_timestamp")
    @classmethod
    def validate_timestamp(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("previous_timestamp must be finite when supplied")
        ####
        return value
    ####

    @field_validator(
        "accelerometer_turn_on_bias",
        "gyroscope_turn_on_bias",
        "accelerometer_bias",
        "gyroscope_bias",
        "previous_velocity",
    )
    @classmethod
    def validate_vectors(cls, value: Vector3 | None) -> Vector3 | None:
        if value is not None and not all(isfinite(component) for component in value):
            raise ValueError("checkpoint vectors must contain only finite values")
        ####
        return value
    ####

    @field_validator("accelerometer_misalignment", "gyroscope_misalignment", "previous_orientation")
    @classmethod
    def validate_matrices(cls, value: Matrix3 | None) -> Matrix3 | None:
        if value is not None and not all(isfinite(component) for row in value for component in row):
            raise ValueError("checkpoint matrices must contain only finite values")
        ####
        return value
    ####

    @field_validator("accelerometer_flicker_states", "gyroscope_flicker_states")
    @classmethod
    def validate_flicker_states(cls, value: FlickerStates) -> FlickerStates:
        if not all(isfinite(component) for row in value for component in row):
            raise ValueError("checkpoint flicker states must contain only finite values")
        ####
        return value
    ####
####
