import numpy as np

from .config import ImuConfig
from .distortions import apply_distortions, clip, quantize
from .interfaces import BaseImuModel
from .kinematics import rotation_vector_from_matrix, validate_orientation
from .processes import FlickerBiasProcess, gauss_markov, white_noise
from .signals import ImuOutput
from .thermal import LinearThermalModel, ThermalModel


class ImuModel(BaseImuModel):
    """Transform truth states into accelerometer and gyro increments."""

    def __init__(self, config: ImuConfig | None = None, rng: np.random.Generator | None = None, thermal_model: ThermalModel | None = None):
        self.config = config or ImuConfig()
        self.rng = rng or np.random.default_rng()
        self.thermal_model = thermal_model or LinearThermalModel()
        self._accel_flicker = FlickerBiasProcess(self.config.accelerometer)
        self._gyro_flicker = FlickerBiasProcess(self.config.gyroscope)
        self.reset()

    def reset(self) -> None:
        # A turn-on bias is fixed for one run. The separate bias states below
        # carry the in-run Gauss-Markov/random-walk process.
        self._accel_turn_on_bias = self.rng.normal(0, self.config.accelerometer.turn_on_bias_std, 3)
        self._gyro_turn_on_bias = self.rng.normal(0, self.config.gyroscope.turn_on_bias_std, 3)
        self._accel_bias = self.rng.normal(0, self.config.accelerometer.bias_std, 3)
        self._gyro_bias = self.rng.normal(0, self.config.gyroscope.bias_std, 3)
        self._accel_flicker.reset(self.rng)
        self._gyro_flicker.reset(self.rng)
        self._accel_misalignment = self._sample_misalignment(self.config.accelerometer.misalignment_std)
        self._gyro_misalignment = self._sample_misalignment(self.config.gyroscope.misalignment_std)
        self._previous_timestamp = None
        self._previous_velocity = None
        self._previous_orientation = None

    def _sample_misalignment(self, std: float) -> np.ndarray:
        vector = self.rng.normal(0.0, std, 3)
        angle = np.linalg.norm(vector)
        if angle == 0:
            return np.eye(3)
        axis = vector / angle
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    def measure(self, timestamp: float, velocity_without_gravity: np.ndarray, orientation_world_from_body: np.ndarray, temperature: float = 25.0) -> ImuOutput:
        """Consume truth velocity and orientation; return deltas over the sample period."""
        if not np.isfinite(timestamp) or not np.isfinite(temperature):
            raise ValueError("timestamp and temperature must be finite")
        velocity = np.asarray(velocity_without_gravity, dtype=float)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_without_gravity must be a finite vector with shape (3,)")
        orientation = validate_orientation(orientation_world_from_body)
        if self._previous_timestamp is None:
            self._previous_timestamp, self._previous_velocity, self._previous_orientation = timestamp, velocity.copy(), orientation.copy()
            return ImuOutput(np.zeros(3), np.zeros(3), 0.0, temperature)
        dt = timestamp - self._previous_timestamp
        if dt <= 0:
            raise ValueError("timestamp must increase")
        a_cfg, g_cfg = self.config.accelerometer, self.config.gyroscope
        true_delta_v = self._previous_orientation.T @ (velocity - self._previous_velocity)
        true_delta_theta = rotation_vector_from_matrix(self._previous_orientation.T @ orientation)
        self._accel_bias = gauss_markov(self._accel_bias, a_cfg.bias_std, a_cfg.bias_correlation_time, dt, self.rng)
        self._gyro_bias = gauss_markov(self._gyro_bias, g_cfg.bias_std, g_cfg.bias_correlation_time, dt, self.rng)
        accel_flicker = self._accel_flicker.step(dt, self.rng)
        gyro_flicker = self._gyro_flicker.step(dt, self.rng)
        a_thermal = self.thermal_model.evaluate(a_cfg, temperature)
        g_thermal = self.thermal_model.evaluate(g_cfg, temperature)
        accel = self._accel_misalignment @ apply_distortions(true_delta_v / dt, a_cfg, a_thermal)
        gyro = self._gyro_misalignment @ apply_distortions(true_delta_theta / dt, g_cfg, g_thermal)
        accel_covariance = self._scale_covariance(a_cfg.noise_covariance, a_thermal.noise_multiplier)
        gyro_covariance = self._scale_covariance(g_cfg.noise_covariance, g_thermal.noise_multiplier)
        accel_noise = white_noise(a_cfg.white_noise_density * a_thermal.noise_multiplier, dt, self.rng, accel_covariance)
        gyro_noise = white_noise(g_cfg.white_noise_density * g_thermal.noise_multiplier, dt, self.rng, gyro_covariance)
        accel = accel + self._accel_turn_on_bias + self._accel_bias + accel_flicker + accel_noise
        gyro = gyro + self._gyro_turn_on_bias + self._gyro_bias + gyro_flicker + gyro_noise
        accel = clip(accel, a_cfg.measurement_range) if a_cfg.apply_clipping else accel
        gyro = clip(gyro, g_cfg.measurement_range) if g_cfg.apply_clipping else gyro
        accel = quantize(accel, a_cfg.quantization_step)
        gyro = quantize(gyro, g_cfg.quantization_step)
        self._previous_timestamp, self._previous_velocity, self._previous_orientation = timestamp, velocity.copy(), orientation.copy()
        return ImuOutput(
            accel * dt * a_cfg.output_scale * self.config.output_scale_accelerometer,
            gyro * dt * g_cfg.output_scale * self.config.output_scale_gyroscope,
            dt,
            temperature,
        )

    @staticmethod
    def _scale_covariance(covariance: tuple[tuple[float, float, float], ...] | None, multiplier: float | np.ndarray) -> np.ndarray | None:
        if covariance is None:
            return None
        matrix = np.asarray(covariance, dtype=float)
        factors = np.asarray(multiplier, dtype=float)
        if factors.ndim == 0:
            return matrix * float(factors) ** 2
        return factors[:, None] * matrix * factors[None, :]
