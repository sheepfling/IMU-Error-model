from numpy import all, array, asarray, cos, eye, isfinite, ndarray, random, sin, zeros
from numpy.linalg import norm

from .config import ImuConfig
from .distortions import apply_distortions, clip, quantize
from .kinematics import rotation_vector_from_matrix, validate_orientation
from .processes import FlickerBiasProcess, gauss_markov, white_noise
from .runtime import CompiledAxisConfig, compile_axis_config
from .signals import ImuOutput
from .thermal import LinearThermalModel, ThermalModel, ThermalState


class ImuModel:
    """Transform truth states into imperfect accelerometer and gyro increments.

    Truth velocity is supplied without gravity in world axes. Outputs are
    expressed in the body axes at the beginning of each sampled interval.
    """

    def __init__(self, config: ImuConfig | None = None, rng: random.Generator | None = None,
                 thermal_model: ThermalModel | None = None):
        self.config = config or ImuConfig()
        self.rng = rng or random.default_rng()
        self._accel_runtime: CompiledAxisConfig = compile_axis_config(self.config.accelerometer)
        self._gyro_runtime: CompiledAxisConfig = compile_axis_config(self.config.gyroscope)
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

    def measure(self, timestamp: float, velocity_without_gravity: ndarray, orientation_world_from_body: ndarray,
                temperature_celsius: float | None = None) -> ImuOutput:
        """Sample the model at `timestamp` and return body-frame increments.

        `velocity_without_gravity` is a world-frame velocity in m/s and
        `orientation_world_from_body` maps body-frame vectors into world
        axes. `temperature_celsius` is optional; when omitted, thermal
        corrections are skipped for this sample.
        """
        if not isfinite(timestamp) or (temperature_celsius is not None and not isfinite(temperature_celsius)):
            raise ValueError("timestamp and temperature_celsius must be finite when supplied")
        ####
        velocity = asarray(velocity_without_gravity, dtype=float)
        if velocity.shape != (3,) or not all(isfinite(velocity)):
            raise ValueError("velocity_without_gravity must be a finite vector with shape (3,)")
        ####
        orientation = validate_orientation(orientation_world_from_body)
        if self._previous_timestamp is None:
            self._previous_timestamp, self._previous_velocity, self._previous_orientation = timestamp, velocity.copy(), orientation.copy()
            return ImuOutput(zeros(3), zeros(3), timestamp, timestamp, temperature_celsius)
        ####
        assert self._previous_velocity is not None
        assert self._previous_orientation is not None
        start_time = self._previous_timestamp
        dt = timestamp - self._previous_timestamp
        if dt <= 0:
            raise ValueError("timestamp must increase")
        ####
        a_cfg, g_cfg = self.config.accelerometer, self.config.gyroscope
        a_runtime, g_runtime = self._accel_runtime, self._gyro_runtime
        true_delta_v = self._previous_orientation.T @ (velocity - self._previous_velocity)
        true_delta_theta = rotation_vector_from_matrix(self._previous_orientation.T @ orientation)
        self._accel_bias = gauss_markov(self._accel_bias, a_runtime.bias_std, a_cfg.bias_correlation_time, dt, self.rng)
        self._gyro_bias = gauss_markov(self._gyro_bias, g_runtime.bias_std, g_cfg.bias_correlation_time, dt, self.rng)
        accel_flicker = self._accel_flicker.step(dt, self.rng)
        gyro_flicker = self._gyro_flicker.step(dt, self.rng)
        if temperature_celsius is None:
            a_thermal = g_thermal = self._identity_thermal
        else:
            a_thermal = self._accel_thermal_model.evaluate(temperature_celsius)
            g_thermal = self._gyro_thermal_model.evaluate(temperature_celsius)
        ####
        accel = self._accel_misalignment @ apply_distortions(true_delta_v / dt, a_cfg, a_thermal,
                                                             a_runtime.scale_factor)
        gyro = self._gyro_misalignment @ apply_distortions(true_delta_theta / dt, g_cfg, g_thermal,
                                                           g_runtime.scale_factor)
        accel_covariance = self._scale_covariance(a_runtime.noise_covariance, a_thermal.noise_multiplier)
        gyro_covariance = self._scale_covariance(g_runtime.noise_covariance, g_thermal.noise_multiplier)
        accel_noise = white_noise(a_cfg.white_noise_density * a_thermal.noise_multiplier, dt, self.rng,
                                  accel_covariance)
        gyro_noise = white_noise(g_cfg.white_noise_density * g_thermal.noise_multiplier, dt, self.rng, gyro_covariance)
        accel = accel + self._accel_turn_on_bias + self._accel_bias + accel_flicker + accel_noise
        gyro = gyro + self._gyro_turn_on_bias + self._gyro_bias + gyro_flicker + gyro_noise
        accel = clip(accel, a_cfg.measurement_range) if a_cfg.apply_clipping else accel
        gyro = clip(gyro, g_cfg.measurement_range) if g_cfg.apply_clipping else gyro
        accel = quantize(accel, a_runtime.quantization_step)
        gyro = quantize(gyro, g_runtime.quantization_step)
        self._previous_timestamp, self._previous_velocity, self._previous_orientation = timestamp, velocity.copy(), orientation.copy()
        return ImuOutput(
            accel * dt * a_cfg.output_scale * self.config.output_scale_accelerometer,
            gyro * dt * g_cfg.output_scale * self.config.output_scale_gyroscope,
            start_time,
            timestamp,
            temperature_celsius,
        )
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
