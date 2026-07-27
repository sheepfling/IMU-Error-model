"""Stateless, estimator-facing characterization of configured IMU errors."""

from collections.abc import Sequence
from dataclasses import dataclass

from numpy import asarray, diag, eye, full, isfinite, ndarray, zeros

from .config import AxisConfig, AxisValue, ImuConfig
from .processes import (
    flicker_component_parameters,
    flicker_transition_parameters,
    gauss_markov_parameters,
)
from .thermal import LinearThermalModel, ThermalState


@dataclass(frozen=True, slots=True)
class PersistentProcessEstimate:
    """Discrete statistics for one channel's persistent error state.

    The state is arranged as three-axis blocks in ``state_block_labels``. The
    ``transition`` and ``process_covariance`` describe one future interval of
    length ``dt``. ``start_mean``/``start_covariance`` and
    ``end_mean``/``end_covariance`` describe the state before and after that
    interval without sampling or mutating a model.
    """

    state_block_labels: tuple[str, ...]
    state_block_slices: tuple[slice, ...]
    transition: ndarray
    process_covariance: ndarray
    initial_mean: ndarray
    initial_covariance: ndarray
    start_mean: ndarray
    start_covariance: ndarray
    end_mean: ndarray
    end_covariance: ndarray
    measurement_matrix: ndarray

    @property
    def prior_bias_mean(self) -> ndarray:
        """Return the end-of-interval mean of the additive persistent error."""
        return self.measurement_matrix @ self.end_mean
    ####

    @property
    def prior_bias_covariance(self) -> ndarray:
        """Return the end-of-interval covariance of the additive persistent error."""
        return self.measurement_matrix @ self.end_covariance @ self.measurement_matrix.T
    ####

    @property
    def initial_bias_mean(self) -> ndarray:
        """Return the additive persistent rate-error mean at reset."""
        return self.measurement_matrix @ self.initial_mean
    ####

    @property
    def initial_bias_covariance(self) -> ndarray:
        """Return the additive persistent rate-error covariance at reset."""
        return self.measurement_matrix @ self.initial_covariance @ self.measurement_matrix.T
    ####

    @property
    def prior_mean(self) -> ndarray:
        """Backward-compatible alias for the end-of-interval prior mean."""
        return self.end_mean
    ####

    @property
    def prior_covariance(self) -> ndarray:
        """Backward-compatible alias for the end-of-interval prior covariance."""
        return self.end_covariance
    ####

    @property
    def bias_transition(self) -> ndarray:
        """Return the transition for the three-axis in-run bias block."""
        if "in_run_bias" not in self.state_block_labels:
            return zeros((0, 0))
        ####
        state_slice = self.state_block_slices[self.state_block_labels.index("in_run_bias")]
        return self.transition[state_slice, state_slice]
    ####

    @property
    def bias_process_covariance(self) -> ndarray:
        """Return process covariance for the three-axis in-run bias block."""
        if "in_run_bias" not in self.state_block_labels:
            return zeros((0, 0))
        ####
        state_slice = self.state_block_slices[self.state_block_labels.index("in_run_bias")]
        return self.process_covariance[state_slice, state_slice]
    ####
####


@dataclass(frozen=True, slots=True)
class ImuChannelErrorProcessEstimate:
    """Estimator-facing process statistics for one IMU channel."""

    continuous_white_noise_covariance: ndarray
    interval_rate_covariance: ndarray
    direct_increment_covariance: ndarray
    conditional_increment_covariance: ndarray
    marginal_increment_covariance: ndarray
    increment_state_process_cross_covariance: ndarray
    deterministic_rate_offset: ndarray
    deterministic_increment_mean: ndarray
    output_scale: float
    persistent: PersistentProcessEstimate
    metadata: tuple[str, ...]

    @property
    def start_conditioned_increment_covariance(self) -> ndarray:
        """Return covariance conditioned on the start state."""
        return self.conditional_increment_covariance
    ####
####


@dataclass(frozen=True, slots=True)
class ImuErrorProcessEstimate:
    """Pure characterization of accelerometer and gyro error statistics.

    The six-dimensional covariance ordering is
    ``[delta_v_x, delta_v_y, delta_v_z, delta_theta_x, delta_theta_y,
    delta_theta_z]``. Persistent means and covariances are priors, not realized
    current bias estimates. The persistent state used for the interval is the
    end-of-interval state after applying the requested transition.
    """

    dt: float
    process_age_at_interval_start_s: float
    accelerometer: ImuChannelErrorProcessEstimate
    gyroscope: ImuChannelErrorProcessEstimate
    direct_increment_covariance: ndarray
    conditional_increment_covariance: ndarray
    marginal_increment_covariance: ndarray
    increment_state_process_cross_covariance: ndarray
    increment_order: tuple[str, ...]
    metadata: tuple[str, ...]

    @property
    def time_since_reset(self) -> float:
        """Backward-compatible alias for the interval-start process age."""
        return self.process_age_at_interval_start_s
    ####
####


def _validate_inputs(
        dt: float,
        process_age_at_interval_start_s: float,
        temperature_celsius: float | None,
) -> None:
    if not isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    ####
    if not isfinite(process_age_at_interval_start_s) or process_age_at_interval_start_s < 0:
        raise ValueError("process_age_at_interval_start_s must be finite and non-negative")
    ####
    if temperature_celsius is not None and not isfinite(temperature_celsius):
        raise ValueError("temperature_celsius must be finite when supplied")
    ####
####


def _as_vector3(value: AxisValue | ndarray) -> ndarray:
    values = asarray(value, dtype=float)
    return full(3, float(values)) if values.ndim == 0 else values
####


def _thermal_state(config: AxisConfig, temperature_celsius: float | None) -> ThermalState:
    if temperature_celsius is None:
        return ThermalState()
    ####
    return LinearThermalModel(config).evaluate(temperature_celsius)
####


def _block_diagonal(blocks: Sequence[ndarray]) -> ndarray:
    dimension = sum(block.shape[0] for block in blocks)
    result = zeros((dimension, dimension))
    start = 0
    for block in blocks:
        size = block.shape[0]
        result[start:start + size, start:start + size] = block
        start += size
    ####
    return result
####


def _persistent_estimate(
        config: AxisConfig,
        dt: float,
        process_age_at_interval_start_s: float,
) -> PersistentProcessEstimate:
    block_labels: list[str] = []
    state_block_slices: list[slice] = []
    taus, flicker_stds = flicker_component_parameters(config)
    transition_blocks: list[ndarray] = []
    process_blocks: list[ndarray] = []
    initial_blocks: list[ndarray] = []
    start_blocks: list[ndarray] = []
    end_blocks: list[ndarray] = []

    def add_block(
            label: str,
            transition_block: ndarray,
            process_block: ndarray,
            initial_block: ndarray,
            start_block: ndarray,
            end_block: ndarray,
    ) -> None:
        block_labels.append(label)
        offset = 3 * len(block_labels) - 3
        state_block_slices.append(slice(offset, offset + 3))
        transition_blocks.append(transition_block)
        process_blocks.append(process_block)
        initial_blocks.append(initial_block)
        start_blocks.append(start_block)
        end_blocks.append(end_block)
    ####

    if bool(asarray(config.turn_on_bias_std).any()):
        turn_on_covariance = diag(_as_vector3(config.turn_on_bias_std) ** 2)
        add_block("turn_on_bias", eye(3), zeros((3, 3)), turn_on_covariance,
                  turn_on_covariance, turn_on_covariance)
    ####

    if bool(asarray(config.bias_std).any()):
        if config.bias_correlation_time is None:
            bias_transition = eye(3)
            bias_process = diag(_as_vector3(config.bias_std) ** 2) * dt
            bias_initial = zeros((3, 3))
            bias_start = diag(_as_vector3(config.bias_std) ** 2) * process_age_at_interval_start_s
            bias_end = bias_start + bias_process
        else:
            bias_phi, bias_variance = gauss_markov_parameters(
                config.bias_std, config.bias_correlation_time, dt
            )
            bias_transition = eye(3) * bias_phi
            bias_process = diag(_as_vector3(bias_variance))
            bias_initial = diag(_as_vector3(config.bias_std) ** 2)
            bias_start = bias_initial.copy()
            bias_end = bias_transition @ bias_start @ bias_transition.T + bias_process
        ####
        add_block("in_run_bias", bias_transition, bias_process, bias_initial, bias_start, bias_end)
    ####

    flicker_phi, flicker_variances = flicker_transition_parameters(taus, flicker_stds, dt)
    elapsed_flicker_phi, elapsed_flicker_variances = flicker_transition_parameters(
        taus, flicker_stds, process_age_at_interval_start_s
    )
    for index, (std, phi, variance, elapsed_phi, elapsed_variance) in enumerate(zip(
            flicker_stds, flicker_phi, flicker_variances,
            elapsed_flicker_phi, elapsed_flicker_variances,
    )):
        transition = eye(3) * phi
        process = eye(3) * variance
        initial = eye(3) * std ** 2
        start = eye(3) * (elapsed_phi ** 2 * std ** 2 + elapsed_variance)
        end = transition @ start @ transition.T + process
        add_block(f"flicker_{index}", transition, process, initial, start, end)
    ####

    initial_mean = zeros(3 * len(block_labels))
    start_mean = zeros(initial_mean.shape)
    end_mean = zeros(initial_mean.shape)
    initial_covariance = _block_diagonal(initial_blocks)
    start_covariance = _block_diagonal(start_blocks)
    end_covariance = _block_diagonal(end_blocks)
    measurement_matrix = zeros((3, initial_mean.size))
    for state_slice in state_block_slices:
        measurement_matrix[:, state_slice] = eye(3)
    ####

    return PersistentProcessEstimate(
        state_block_labels=tuple(block_labels),
        state_block_slices=tuple(state_block_slices),
        transition=_block_diagonal(transition_blocks),
        process_covariance=_block_diagonal(process_blocks),
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        start_mean=start_mean,
        start_covariance=start_covariance,
        end_mean=end_mean,
        end_covariance=end_covariance,
        measurement_matrix=measurement_matrix,
    )
####


def _channel_estimate(
        config: AxisConfig,
        global_output_scale: float,
        dt: float,
        process_age_at_interval_start_s: float,
        temperature_celsius: float | None,
        include_quantization_approximation: bool,
) -> ImuChannelErrorProcessEstimate:
    thermal = _thermal_state(config, temperature_celsius)
    multiplier = _as_vector3(thermal.noise_multiplier)
    multiplier_matrix = diag(multiplier)
    if config.noise_covariance is None:
        continuous_covariance = eye(3) * config.white_noise_density ** 2
    else:
        continuous_covariance = asarray(config.noise_covariance, dtype=float)
    ####
    interval_rate_covariance = (
            multiplier_matrix @ continuous_covariance @ multiplier_matrix.T / dt
    )
    output_scale = config.output_scale * global_output_scale
    scale_matrix = eye(3) * output_scale
    direct_increment_covariance = (
            dt ** 2 * scale_matrix @ interval_rate_covariance @ scale_matrix.T
    )

    metadata: list[str] = [
        "Scale factor, radial nonlinearity, and misalignment covariance are omitted because no nominal input rate is supplied.",
        "Clipping is omitted because it is signal-dependent and nonlinear.",
        "Persistent state statistics are priors; realized turn-on and current bias values are not available.",
        "Thermal characterization uses the built-in LinearThermalModel; custom ThermalModel instances supplied to ImuModel are outside this config-only API.",
    ]
    conditional_increment_covariance = direct_increment_covariance.copy()
    if include_quantization_approximation and config.quantization_step is not None:
        quantization_rate_covariance = diag(_as_vector3(config.quantization_step) ** 2 / 12.0)
        conditional_increment_covariance += (
                dt ** 2 * scale_matrix @ quantization_rate_covariance @ scale_matrix.T
        )
        metadata.append("Quantization uses the high-resolution uniform-noise approximation delta^2/12.")
    elif config.quantization_step is not None:
        metadata.append("Exact quantization is omitted from covariance characterization by request.")
    ####

    persistent = _persistent_estimate(config, dt, process_age_at_interval_start_s)
    state_to_increment = dt * scale_matrix @ persistent.measurement_matrix
    increment_state_process_cross_covariance = state_to_increment @ persistent.process_covariance
    conditional_increment_covariance = conditional_increment_covariance + (
            state_to_increment @ persistent.process_covariance @ state_to_increment.T
    )
    marginal_increment_covariance = direct_increment_covariance + (
            state_to_increment @ persistent.end_covariance @ state_to_increment.T
    )
    if temperature_celsius is None:
        metadata.append(
            "Temperature-dependent deterministic and noise scaling are omitted because temperature is unavailable.")
    ####

    return ImuChannelErrorProcessEstimate(
        continuous_white_noise_covariance=continuous_covariance,
        interval_rate_covariance=interval_rate_covariance,
        direct_increment_covariance=direct_increment_covariance,
        conditional_increment_covariance=conditional_increment_covariance,
        marginal_increment_covariance=marginal_increment_covariance,
        increment_state_process_cross_covariance=increment_state_process_cross_covariance,
        deterministic_rate_offset=_as_vector3(thermal.bias_offset),
        deterministic_increment_mean=dt * scale_matrix @ _as_vector3(thermal.bias_offset),
        output_scale=output_scale,
        persistent=persistent,
        metadata=tuple(metadata),
    )
####


def characterize_imu_error_process(
        config: ImuConfig,
        *,
        dt: float,
        process_age_at_interval_start_s: float = 0.0,
        temperature_celsius: float | None = None,
        include_quantization_approximation: bool = False,
) -> ImuErrorProcessEstimate:
    """Return deterministic estimator-facing process statistics.

    This function performs no random-number generation and does not inspect or
    mutate an :class:`ImuModel`. It reports configuration priors, not a realized
    current bias estimate. Accelerometer and gyroscope cross-covariance is zero
    because the configuration has no cross-channel noise term.
    """
    _validate_inputs(dt, process_age_at_interval_start_s, temperature_celsius)
    accelerometer = _channel_estimate(
        config.accelerometer,
        config.output_scale_accelerometer,
        dt,
        process_age_at_interval_start_s,
        temperature_celsius,
        include_quantization_approximation,
    )
    gyroscope = _channel_estimate(
        config.gyroscope,
        config.output_scale_gyroscope,
        dt,
        process_age_at_interval_start_s,
        temperature_celsius,
        include_quantization_approximation,
    )
    conditional = zeros((6, 6))
    direct = zeros((6, 6))
    marginal = zeros((6, 6))
    accel_state_size = accelerometer.persistent.measurement_matrix.shape[1]
    gyro_state_size = gyroscope.persistent.measurement_matrix.shape[1]
    increment_state_process_cross_covariance = zeros((6, accel_state_size + gyro_state_size))
    conditional[:3, :3] = accelerometer.conditional_increment_covariance
    conditional[3:, 3:] = gyroscope.conditional_increment_covariance
    direct[:3, :3] = accelerometer.direct_increment_covariance
    direct[3:, 3:] = gyroscope.direct_increment_covariance
    marginal[:3, :3] = accelerometer.marginal_increment_covariance
    marginal[3:, 3:] = gyroscope.marginal_increment_covariance
    increment_state_process_cross_covariance[:3, :accel_state_size] = (
        accelerometer.increment_state_process_cross_covariance
    )
    increment_state_process_cross_covariance[3:, accel_state_size:] = (
        gyroscope.increment_state_process_cross_covariance
    )
    metadata = (
        "Covariance ordering is [delta_v_x, delta_v_y, delta_v_z, delta_theta_x, delta_theta_y, delta_theta_z].",
        "Persistent states are advanced over dt before they are applied to the returned interval.",
        "process_age_at_interval_start_s is modeled time since the first history-initializing sample, not wall-clock reset time.",
        "Accelerometer and gyroscope cross-covariance is zero because no cross-channel covariance is configured.",
        "Marginal increment covariance includes persistent-state prior uncertainty and must not be added as fresh independent noise when the filter carries those states.",
        *accelerometer.metadata,
        *gyroscope.metadata,
    )
    return ImuErrorProcessEstimate(
        dt=dt,
        process_age_at_interval_start_s=process_age_at_interval_start_s,
        accelerometer=accelerometer,
        gyroscope=gyroscope,
        direct_increment_covariance=direct,
        conditional_increment_covariance=conditional,
        marginal_increment_covariance=marginal,
        increment_state_process_cross_covariance=increment_state_process_cross_covariance,
        increment_order=(
            "delta_v_x", "delta_v_y", "delta_v_z",
            "delta_theta_x", "delta_theta_y", "delta_theta_z",
        ),
        metadata=metadata,
    )
####
