from __future__ import annotations

from copy import deepcopy
from os import fsync, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, ClassVar, TypeGuard, cast

from numpy import all, array, asarray, cos, eye, isclose, isfinite, ndarray, random, sin, uint64, zeros
from numpy.linalg import norm

from .checkpoint import ImuModelCheckpoint, JsonValue, Matrix3, RngBitGeneratorName, Vector3
from .checkpoint_codecs import CheckpointCodecProtocol, PydanticJsonCheckpointCodec
from .config import ImuConfig
from .distortions import apply_distortions, clip, quantize
from .kinematics import rotation_vector_from_matrix, validate_orientation
from .processes import FlickerBiasProcess, gauss_markov, white_noise
from .runtime import CompiledAxisConfig, compile_axis_config
from .signals import ImuOutput
from .thermal import LinearThermalModel, ThermalModel, ThermalState


def _vector3_tuple(value: ndarray) -> Vector3:
    values = asarray(value, dtype=float)
    if values.shape != (3,) or not all(isfinite(values)):
        raise ValueError("checkpoint state must be a finite vector with shape (3,)")
    ####
    return float(values[0]), float(values[1]), float(values[2])
####


def _matrix3_tuple(value: ndarray) -> Matrix3:
    values = asarray(value, dtype=float)
    if values.shape != (3, 3) or not all(isfinite(values)):
        raise ValueError("checkpoint state must be a finite matrix with shape (3, 3)")
    ####
    return _vector3_tuple(values[0]), _vector3_tuple(values[1]), _vector3_tuple(values[2])
####


def _flicker_state_tuple(value: ndarray) -> tuple[tuple[float, float, float], ...]:
    values = asarray(value, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not all(isfinite(values)):
        raise ValueError("checkpoint flicker state must be finite with shape (n, 3)")
    ####
    return tuple(_vector3_tuple(row) for row in values)
####


def _array_vector3(value: tuple[float, float, float]) -> ndarray:
    values = asarray(value, dtype=float)
    if values.shape != (3,) or not all(isfinite(values)):
        raise ValueError("checkpoint state must be a finite vector with shape (3,)")
    ####
    return values.copy()
####


def _array_matrix3(value: tuple[tuple[float, float, float], ...]) -> ndarray:
    values = asarray(value, dtype=float)
    if values.shape != (3, 3) or not all(isfinite(values)):
        raise ValueError("checkpoint state must be a finite matrix with shape (3, 3)")
    ####
    return values.copy()
####


def _array_flicker_states(value: tuple[tuple[float, float, float], ...], expected_shape: tuple[int, int]) -> ndarray:
    values = asarray(value, dtype=float)
    if values.size == 0 and expected_shape == (0, 3):
        values = zeros((0, 3))
    ####
    if values.shape != expected_shape or not all(isfinite(values)):
        raise ValueError("checkpoint flicker state has the wrong configured shape")
    ####
    return values.copy()
####


def _checkpoint_rng(checkpoint: ImuModelCheckpoint) -> random.Generator:
    if checkpoint.rng_bit_generator == "PCG64":
        bit_generator = random.PCG64()
    elif checkpoint.rng_bit_generator == "PCG64DXSM":
        bit_generator = random.PCG64DXSM()
    elif checkpoint.rng_bit_generator == "Philox":
        bit_generator = random.Philox()
    elif checkpoint.rng_bit_generator == "SFC64":
        bit_generator = random.SFC64()
    elif checkpoint.rng_bit_generator == "MT19937":
        bit_generator = random.MT19937()
    else:
        raise ValueError(
            f"unsupported checkpoint bit generator: {checkpoint.rng_bit_generator!r}"
        )
    ####
    try:
        cast(Any, bit_generator).state = _restore_json_rng_state(checkpoint.rng_state)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint contains an invalid NumPy RNG state") from error
    ####
    return random.Generator(bit_generator)
####


def _json_compatible(value: Any) -> JsonValue:
    if isinstance(value, ndarray):
        return _json_compatible(value.tolist())
    ####
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    ####
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    ####
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    ####
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    ####
    raise TypeError(f"checkpoint state contains a non-JSON value: {type(value).__name__}")
####


def _restore_json_rng_state(value: JsonValue) -> Any:
    if isinstance(value, dict):
        return {key: _restore_json_rng_state(item) for key, item in value.items()}
    ####
    if isinstance(value, list):
        return asarray([_restore_json_rng_state(item) for item in value], dtype=uint64)
    ####
    return value
####


def _is_supported_bit_generator(value: str) -> TypeGuard[RngBitGeneratorName]:
    """Return whether a NumPy bit-generator name is checkpoint-compatible."""
    return value in {"PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"}
####


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            fsync(temporary.fileno())
        ####
        if temporary_path is None:
            raise RuntimeError("temporary checkpoint file was not created")
        ####
        replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        ####
    ####
####


class ImuModel:
    """Transform truth states into imperfect accelerometer and gyro increments.

    Truth velocity is supplied without gravity in world axes. Outputs are
    expressed in the body axes at the beginning of each sampled interval.
    """

    _default_checkpoint_codec: ClassVar[CheckpointCodecProtocol[ImuModelCheckpoint]] = PydanticJsonCheckpointCodec(
        ImuModelCheckpoint
    )
    checkpoint_codec: CheckpointCodecProtocol[ImuModelCheckpoint]

    def __init__(self, config: ImuConfig | None = None, rng: random.Generator | None = None,
                 thermal_model: ThermalModel | None = None) -> None:
        self.checkpoint_codec = self._default_checkpoint_codec
        self.config = config or ImuConfig()
        self.rng = rng or random.default_rng()
        self._accel_runtime: CompiledAxisConfig = compile_axis_config(self.config.accelerometer)
        self._gyro_runtime: CompiledAxisConfig = compile_axis_config(self.config.gyroscope)
        self._gyro_specific_force_sensitivity = (
            zeros((3, 3))
            if self.config.gyroscope_specific_force_sensitivity is None
            else asarray(self.config.gyroscope_specific_force_sensitivity, dtype=float).copy()
        )
        self._gyro_specific_force_sensitivity.setflags(write=False)
        self._gyro_specific_force_sensitivity_temperature_coefficient = (
            zeros((3, 3))
            if self.config.gyroscope_specific_force_sensitivity_temperature_coefficient is None
            else asarray(
                self.config.gyroscope_specific_force_sensitivity_temperature_coefficient,
                dtype=float,
            ).copy()
        )
        self._gyro_specific_force_sensitivity_temperature_coefficient.setflags(write=False)
        self._identity_thermal = ThermalState()
        self._accel_thermal_model = LinearThermalModel(
            self.config.accelerometer) if thermal_model is None else thermal_model
        self._gyro_thermal_model = LinearThermalModel(self.config.gyroscope) if thermal_model is None else thermal_model
        self._accel_flicker = FlickerBiasProcess(self.config.accelerometer)
        self._gyro_flicker = FlickerBiasProcess(self.config.gyroscope)
        self._accel_turn_on_bias: ndarray = zeros(3)
        self._gyro_turn_on_bias: ndarray = zeros(3)
        self._accel_bias: ndarray = zeros(3)
        self._gyro_bias: ndarray = zeros(3)
        self._accel_misalignment: ndarray = eye(3)
        self._gyro_misalignment: ndarray = eye(3)
        self._previous_timestamp: float | None = None
        self._previous_velocity: ndarray | None = None
        self._previous_orientation: ndarray | None = None
        self.reset()
    ####

    def reset(self) -> None:
        """Reset run-level errors, stochastic states, and the truth baseline."""
        # A turn-on bias is fixed for one run. The separate bias states below
        # carry the in-run Gauss-Markov/random-walk process.
        self._accel_turn_on_bias = self.rng.normal(0, self._accel_runtime.turn_on_bias_std, 3)
        self._gyro_turn_on_bias = self.rng.normal(0, self._gyro_runtime.turn_on_bias_std, 3)
        self._accel_bias = (
            self.rng.normal(0, self._accel_runtime.bias_std, 3)
            if self.config.accelerometer.bias_correlation_time is not None
            else zeros(3)
        )
        self._gyro_bias = (
            self.rng.normal(0, self._gyro_runtime.bias_std, 3)
            if self.config.gyroscope.bias_correlation_time is not None
            else zeros(3)
        )
        self._accel_flicker.reset(self.rng)
        self._gyro_flicker.reset(self.rng)
        self._accel_misalignment = self._sample_misalignment(
            std=self.config.accelerometer.misalignment_std,
            covariance=self._accel_runtime.misalignment_covariance,
            rng=self.rng,
        )
        self._gyro_misalignment = self._sample_misalignment(
            std=self.config.gyroscope.misalignment_std,
            covariance=self._gyro_runtime.misalignment_covariance,
            rng=self.rng,
        )
        self._previous_timestamp = None
        self._previous_velocity = None
        self._previous_orientation = None
    ####

    @classmethod
    def from_checkpoint(cls, checkpoint: ImuModelCheckpoint) -> ImuModel:
        """Construct a model whose next sample resumes a saved checkpoint."""
        if checkpoint.model_type != "imu_error_model.ImuModel":
            raise ValueError(f"unsupported checkpoint model type: {checkpoint.model_type!r}")
        ####
        model = cls(config=checkpoint.config, rng=_checkpoint_rng(checkpoint))
        model.restore(checkpoint)
        return model
    ####

    def snapshot(self) -> ImuModelCheckpoint:
        """Return a versioned, JSON-serializable snapshot of the complete model state."""
        if not isinstance(self._accel_thermal_model, LinearThermalModel) or not isinstance(
                self._gyro_thermal_model, LinearThermalModel
        ):
            raise TypeError("checkpointing requires the built-in LinearThermalModel")
        ####
        bit_generator = type(self.rng.bit_generator).__name__
        if not _is_supported_bit_generator(bit_generator):
            raise TypeError(f"checkpointing does not support NumPy bit generator {bit_generator!r}")
        ####
        rng_state = _json_compatible(self.rng.bit_generator.state)
        if not isinstance(rng_state, dict):
            raise TypeError("NumPy RNG state must be a JSON object")
        ####
        return ImuModelCheckpoint(
            config=self.config,
            rng_bit_generator=bit_generator,
            rng_state=deepcopy(rng_state),
            accelerometer_turn_on_bias=_vector3_tuple(self._accel_turn_on_bias),
            gyroscope_turn_on_bias=_vector3_tuple(self._gyro_turn_on_bias),
            accelerometer_bias=_vector3_tuple(self._accel_bias),
            gyroscope_bias=_vector3_tuple(self._gyro_bias),
            accelerometer_flicker_states=_flicker_state_tuple(self._accel_flicker.snapshot()),
            gyroscope_flicker_states=_flicker_state_tuple(self._gyro_flicker.snapshot()),
            accelerometer_misalignment=_matrix3_tuple(self._accel_misalignment),
            gyroscope_misalignment=_matrix3_tuple(self._gyro_misalignment),
            previous_timestamp=self._previous_timestamp,
            previous_velocity=(
                None if self._previous_velocity is None else _vector3_tuple(self._previous_velocity)
            ),
            previous_orientation=(
                None if self._previous_orientation is None else _matrix3_tuple(self._previous_orientation)
            ),
        )
    ####

    def restore(self, checkpoint: ImuModelCheckpoint) -> None:
        """Replace the model state with a previously captured checkpoint."""
        if checkpoint.schema_version != 1:
            raise ValueError(f"unsupported checkpoint schema version: {checkpoint.schema_version}")
        ####
        if checkpoint.model_type != "imu_error_model.ImuModel":
            raise ValueError(f"unsupported checkpoint model type: {checkpoint.model_type!r}")
        ####
        if checkpoint.config != self.config:
            raise ValueError("checkpoint configuration does not match this model")
        ####
        if checkpoint.thermal_model_type != "linear":
            raise ValueError(f"unsupported checkpoint thermal model: {checkpoint.thermal_model_type!r}")
        ####
        if not isinstance(self._accel_thermal_model, LinearThermalModel) or not isinstance(
                self._gyro_thermal_model, LinearThermalModel
        ):
            raise TypeError("checkpointing requires the built-in LinearThermalModel")
        ####
        restored_rng = _checkpoint_rng(checkpoint)
        accel_turn_on_bias = _array_vector3(checkpoint.accelerometer_turn_on_bias)
        gyro_turn_on_bias = _array_vector3(checkpoint.gyroscope_turn_on_bias)
        accel_bias = _array_vector3(checkpoint.accelerometer_bias)
        gyro_bias = _array_vector3(checkpoint.gyroscope_bias)
        accel_flicker_states = _array_flicker_states(
            checkpoint.accelerometer_flicker_states,
            self._accel_flicker.snapshot().shape,
        )
        gyro_flicker_states = _array_flicker_states(
            checkpoint.gyroscope_flicker_states,
            self._gyro_flicker.snapshot().shape,
        )
        accel_misalignment = _array_matrix3(checkpoint.accelerometer_misalignment)
        gyro_misalignment = _array_matrix3(checkpoint.gyroscope_misalignment)
        previous_velocity = (
            None if checkpoint.previous_velocity is None else _array_vector3(checkpoint.previous_velocity)
        )
        previous_orientation = (
            None if checkpoint.previous_orientation is None else _array_matrix3(checkpoint.previous_orientation)
        )
        self.rng = restored_rng
        self._accel_turn_on_bias = accel_turn_on_bias
        self._gyro_turn_on_bias = gyro_turn_on_bias
        self._accel_bias = accel_bias
        self._gyro_bias = gyro_bias
        self._accel_flicker.restore(accel_flicker_states)
        self._gyro_flicker.restore(gyro_flicker_states)
        self._accel_misalignment = accel_misalignment
        self._gyro_misalignment = gyro_misalignment
        self._previous_timestamp = checkpoint.previous_timestamp
        self._previous_velocity = previous_velocity
        self._previous_orientation = previous_orientation
    ####

    def save_checkpoint(
            self,
            path: str | Path,
            *,
            codec: CheckpointCodecProtocol[ImuModelCheckpoint] | None = None,
    ) -> None:
        """Write the complete model state atomically through a selected codec."""
        destination = Path(path)
        active_codec = self.checkpoint_codec if codec is None else codec
        _atomic_write_bytes(destination, active_codec.encode(self.snapshot()))
    ####

    @classmethod
    def load_checkpoint(
            cls,
            path: str | Path,
            *,
            codec: CheckpointCodecProtocol[ImuModelCheckpoint] | None = None,
    ) -> ImuModel:
        """Load a checkpoint through a selected codec and resume it."""
        active_codec = cls._default_checkpoint_codec if codec is None else codec
        checkpoint = active_codec.decode(Path(path).read_bytes())
        if not isinstance(checkpoint, ImuModelCheckpoint):
            raise TypeError("checkpoint codec returned an unexpected checkpoint type")
        ####
        return cls.from_checkpoint(checkpoint)
    ####

    def measure(self, timestamp: float, velocity_without_gravity: ndarray, orientation_world_from_body: ndarray,
                temperature_celsius: float | None = None) -> ImuOutput:
        """Sample a gravity-excluded truth trajectory and return body-frame increments.

        `velocity_without_gravity` is a world-frame velocity in m/s and
        `orientation_world_from_body` maps body-frame vectors into world
        axes. `temperature_celsius` is optional; when omitted, thermal
        corrections are skipped for this sample.
        """
        self._validate_time_and_temperature(timestamp, temperature_celsius)
        velocity = self._validated_vector3(velocity_without_gravity, "velocity_without_gravity")
        orientation = validate_orientation(orientation_world_from_body)
        if (
                self._previous_timestamp is None
                or self._previous_velocity is None
                or self._previous_orientation is None
        ):
            if self._previous_timestamp is not None and timestamp < self._previous_timestamp:
                raise ValueError("timestamp must not precede the previous measurement boundary")
            ####
            self._previous_timestamp = timestamp
            self._previous_velocity = velocity.copy()
            self._previous_orientation = orientation.copy()
            return ImuOutput(zeros(3), zeros(3), timestamp, timestamp, temperature_celsius)
        ####
        start_time = self._previous_timestamp
        dt = timestamp - start_time
        if dt <= 0:
            raise ValueError("timestamp must increase")
        ####
        true_delta_v = self._previous_orientation.T @ (velocity - self._previous_velocity)
        true_delta_theta = rotation_vector_from_matrix(self._previous_orientation.T @ orientation)
        output = self._corrupt_ideal_rates(
            start_time=start_time,
            end_time=timestamp,
            specific_force_body=true_delta_v / dt,
            angular_rate_body=true_delta_theta / dt,
            temperature_celsius=temperature_celsius,
        )
        self._previous_timestamp = timestamp
        self._previous_velocity = velocity.copy()
        self._previous_orientation = orientation.copy()
        return output
    ####

    def measure_ideal_rates(
            self,
            *,
            start_time: float,
            end_time: float,
            specific_force_body: ndarray,
            angular_rate_body: ndarray,
            temperature_celsius: float | None = None,
    ) -> ImuOutput:
        """Corrupt externally supplied ideal body-frame rates over one interval.

        This lower-level entry point is intended for calibration experiments,
        hardware replay, and reference generators that already own gravity,
        Earth-rate, coning, or sculling semantics. Intervals must be contiguous
        within a run. Calling this method clears the truth-trajectory baseline;
        a later `measure` call establishes a new baseline before returning data.
        """
        self._validate_interval(start_time, end_time, temperature_celsius)
        if self._previous_timestamp is not None and not bool(
                isclose(start_time, self._previous_timestamp, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("start_time must equal the previous measurement boundary; call reset() for a new run")
        ####
        specific_force = self._validated_vector3(specific_force_body, "specific_force_body")
        angular_rate = self._validated_vector3(angular_rate_body, "angular_rate_body")
        output = self._corrupt_ideal_rates(
            start_time=start_time,
            end_time=end_time,
            specific_force_body=specific_force,
            angular_rate_body=angular_rate,
            temperature_celsius=temperature_celsius,
        )
        self._previous_timestamp = end_time
        self._previous_velocity = None
        self._previous_orientation = None
        return output
    ####

    def measure_ideal_increments(
            self,
            *,
            start_time: float,
            end_time: float,
            delta_v_body: ndarray,
            delta_theta_body: ndarray,
            temperature_celsius: float | None = None,
    ) -> ImuOutput:
        """Corrupt externally supplied ideal body-frame increments over one interval."""
        self._validate_interval(start_time, end_time, temperature_celsius)
        delta_v = self._validated_vector3(delta_v_body, "delta_v_body")
        delta_theta = self._validated_vector3(delta_theta_body, "delta_theta_body")
        dt = end_time - start_time
        return self.measure_ideal_rates(
            start_time=start_time,
            end_time=end_time,
            specific_force_body=delta_v / dt,
            angular_rate_body=delta_theta / dt,
            temperature_celsius=temperature_celsius,
        )
    ####

    def _corrupt_ideal_rates(
            self,
            *,
            start_time: float,
            end_time: float,
            specific_force_body: ndarray,
            angular_rate_body: ndarray,
            temperature_celsius: float | None,
    ) -> ImuOutput:
        """Apply the configured deterministic and stochastic channel pipeline."""
        dt = end_time - start_time
        a_cfg, g_cfg = self.config.accelerometer, self.config.gyroscope
        a_runtime, g_runtime = self._accel_runtime, self._gyro_runtime
        self._accel_bias = gauss_markov(
            self._accel_bias, a_runtime.bias_std, a_cfg.bias_correlation_time, dt, self.rng
        )
        self._gyro_bias = gauss_markov(
            self._gyro_bias, g_runtime.bias_std, g_cfg.bias_correlation_time, dt, self.rng
        )
        accel_flicker = self._accel_flicker.step(dt, self.rng)
        gyro_flicker = self._gyro_flicker.step(dt, self.rng)
        if temperature_celsius is None:
            a_thermal = g_thermal = self._identity_thermal
        else:
            a_thermal = self._accel_thermal_model.evaluate(temperature_celsius)
            g_thermal = self._gyro_thermal_model.evaluate(temperature_celsius)
        ####
        accel_intrinsic = apply_distortions(
            specific_force_body,
            a_cfg,
            a_thermal,
            a_runtime.scale_factor,
            a_runtime.fixed_bias,
            a_runtime.sensitivity_matrix,
        )
        gyro_intrinsic = apply_distortions(
            angular_rate_body,
            g_cfg,
            g_thermal,
            g_runtime.scale_factor,
            g_runtime.fixed_bias,
            g_runtime.sensitivity_matrix,
        )
        gyro_specific_force_sensitivity = self._gyro_specific_force_sensitivity
        if temperature_celsius is not None:
            delta_temperature = temperature_celsius - g_cfg.reference_temperature_celsius
            gyro_specific_force_sensitivity = (
                gyro_specific_force_sensitivity
                + delta_temperature * self._gyro_specific_force_sensitivity_temperature_coefficient
            )
        ####
        gyro_intrinsic = gyro_intrinsic + gyro_specific_force_sensitivity @ specific_force_body
        accel = self._accel_misalignment @ accel_intrinsic
        gyro = self._gyro_misalignment @ gyro_intrinsic
        accel_covariance = self._scale_covariance(a_runtime.noise_covariance, a_thermal.noise_multiplier)
        gyro_covariance = self._scale_covariance(g_runtime.noise_covariance, g_thermal.noise_multiplier)
        accel_noise = white_noise(
            a_cfg.white_noise_density * a_thermal.noise_multiplier, dt, self.rng, accel_covariance
        )
        gyro_noise = white_noise(
            g_cfg.white_noise_density * g_thermal.noise_multiplier, dt, self.rng, gyro_covariance
        )
        accel = accel + self._accel_turn_on_bias + self._accel_bias + accel_flicker + accel_noise
        gyro = gyro + self._gyro_turn_on_bias + self._gyro_bias + gyro_flicker + gyro_noise
        accel = clip(accel, a_cfg.measurement_range) if a_cfg.apply_clipping else accel
        gyro = clip(gyro, g_cfg.measurement_range) if g_cfg.apply_clipping else gyro
        accel = quantize(accel, a_runtime.quantization_step)
        gyro = quantize(gyro, g_runtime.quantization_step)
        return ImuOutput(
            accel * dt * a_cfg.output_scale * self.config.output_scale_accelerometer,
            gyro * dt * g_cfg.output_scale * self.config.output_scale_gyroscope,
            start_time,
            end_time,
            temperature_celsius,
        )
    ####

    @staticmethod
    def _validate_time_and_temperature(timestamp: float, temperature_celsius: float | None) -> None:
        if not isfinite(timestamp) or (temperature_celsius is not None and not isfinite(temperature_celsius)):
            raise ValueError("timestamps and temperature_celsius must be finite when supplied")
        ####
    ####

    @classmethod
    def _validate_interval(
            cls, start_time: float, end_time: float, temperature_celsius: float | None
    ) -> None:
        cls._validate_time_and_temperature(start_time, temperature_celsius)
        cls._validate_time_and_temperature(end_time, temperature_celsius)
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        ####
    ####

    @staticmethod
    def _validated_vector3(value: ndarray, field_name: str) -> ndarray:
        vector = asarray(value, dtype=float)
        if vector.shape != (3,) or not all(isfinite(vector)):
            raise ValueError(f"{field_name} must be a finite vector with shape (3,)")
        ####
        return vector
    ####

    @staticmethod
    def _scale_covariance(covariance: ndarray | None,
                          multiplier: float | ndarray) -> ndarray | None:
        if covariance is None:
            return None
        ####
        matrix = asarray(covariance, dtype=float)
        factors = asarray(multiplier, dtype=float)
        if factors.ndim == 0:
            return matrix * float(factors) ** 2
        ####
        return factors[:, None] * matrix * factors[None, :]
    ####

    @staticmethod
    def _sample_misalignment(
            std: float,
            covariance: ndarray | None,
            rng: random.Generator,
    ) -> ndarray:
        vector = (
            rng.multivariate_normal(zeros(3), asarray(covariance, dtype=float))
            if covariance is not None
            else rng.normal(0.0, std, 3)
        )
        angle = norm(vector)
        if angle <= 0:
            return eye(3)
        ####
        axis = vector / angle
        x, y, z = axis
        skew = array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        if angle < 1e-4:
            sine = angle - angle ** 3 / 6.0 + angle ** 5 / 120.0
            one_minus_cosine = angle ** 2 / 2.0 - angle ** 4 / 24.0 + angle ** 6 / 720.0
        else:
            sine = sin(angle)
            one_minus_cosine = 1.0 - cos(angle)
        ####
        return eye(3) + sine * skew + one_minus_cosine * (skew @ skew)
    ####
####
