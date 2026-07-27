from numpy import all as all_values, array, asarray, exp, expm1, full, geomspace, ndarray, random, sqrt, sum, zeros

from .config import AxisConfig, AxisValue


def white_noise(density: float | ndarray, dt: float, rng: random.Generator,
                covariance: ndarray | None = None) -> ndarray:
    """Sample continuous-time white rate noise over an interval.

    `density` is an amplitude spectral density in sensor units per square
    root second. The sampled standard deviation scales as `1/sqrt(dt)`.
    """
    if dt <= 0:
        return zeros(3)
    ####
    if covariance is not None:
        return rng.multivariate_normal(zeros(3), asarray(covariance) / dt)
    ####
    if bool(all_values(asarray(density) == 0)):
        return zeros(3)
    ####
    return rng.normal(0.0, density / sqrt(dt), 3)
####

def gauss_markov(
        current: ndarray,
        std: AxisValue | ndarray,
        correlation_time: float | None,
        dt: float,
        rng: random.Generator,
) -> ndarray:
    """Advance a stationary first-order Gauss–Markov bias process."""
    if dt <= 0 or bool(all_values(asarray(std) == 0)):
        return current.copy()
    ####
    phi, innovation_variance = gauss_markov_parameters(std, correlation_time, dt)
    return phi * current + rng.normal(0.0, sqrt(innovation_variance), 3)
####


def gauss_markov_parameters(
        std: AxisValue | ndarray,
        correlation_time: float | None,
        dt: float,
) -> tuple[float, float | ndarray]:
    """Return exact transition coefficient and innovation variance.

    With no correlation time, the configured standard deviation is the
    continuous-time random-walk driving standard deviation. ``expm1`` keeps
    the stationary Gauss–Markov variance accurate for very short intervals.
    """
    std_values = asarray(std, dtype=float)
    if correlation_time is None:
        variance = std_values ** 2 * dt
        return 1.0, float(variance) if variance.ndim == 0 else variance
    ####
    phi = exp(-dt / correlation_time)
    innovation_variance = std_values ** 2 * -expm1(-2.0 * dt / correlation_time)
    if innovation_variance.ndim == 0:
        innovation_variance = float(innovation_variance)
    ####
    return phi, innovation_variance
####


def flicker_transition_parameters(
        taus: ndarray,
        stds: ndarray,
        dt: float,
) -> tuple[ndarray, ndarray]:
    """Return exact OU transitions and innovation variances for flicker terms."""
    phi = exp(-dt / taus)
    innovation_variance = stds ** 2 * -expm1(-2.0 * dt / taus)
    return phi, innovation_variance
####


def flicker_component_parameters(config: AxisConfig) -> tuple[ndarray, ndarray]:
    """Return OU correlation times and calibrated component standard deviations.

    The characterization API uses the same finite-band approximation as the
    stateful sampler. Keeping the calibration in one helper prevents the two
    paths from silently describing different flicker processes.
    """
    if config.flicker_bias_std == 0:
        return zeros(0), zeros(0)
    ####
    assert config.flicker_min_correlation_time is not None
    assert config.flicker_max_correlation_time is not None
    taus = geomspace(
        config.flicker_min_correlation_time,
        config.flicker_max_correlation_time,
        config.flicker_components,
    )
    center = float(sqrt(taus[0] * taus[-1]))
    unit_variances = array([
        FlickerBiasProcess.allan_variance(1.0, float(tau), center)
        for tau in taus
    ])
    scale = config.flicker_bias_std / sqrt(sum(unit_variances) / config.flicker_components)
    return taus, full(config.flicker_components, scale / sqrt(config.flicker_components))
####


class FlickerBiasProcess:
    """Finite-band 1/f-like bias from logarithmically spaced OU processes.

    The target Allan deviation is calibrated at the geometric-center averaging
    time of the configured correlation-time band. This finite-band
    approximation is intended to match a datasheet Allan-deviation plateau.
    """

    def __init__(self, config: AxisConfig):
        self._target = config.flicker_bias_std
        self._taus, self._stds = flicker_component_parameters(config)
        self._states = zeros((self._taus.size, 3))
        self._last_dt: float | None = None
        self._last_phi = zeros(0)
        self._last_innovation_std = zeros(0)
    ####

    @staticmethod
    def allan_variance(variance: float, correlation_time: float, averaging_time: float) -> float:
        ratio = averaging_time / correlation_time
        exp_term = exp(-ratio)
        average_variance = 2.0 * variance * (
                correlation_time * averaging_time - correlation_time ** 2 * (1.0 - exp_term)
        ) / averaging_time ** 2
        adjacent_covariance = variance * correlation_time ** 2 * (1.0 - exp_term) ** 2 / averaging_time ** 2
        return average_variance - adjacent_covariance
    ####

    def reset(self, rng: random.Generator | None = None) -> None:
        if rng is None:
            self._states.fill(0.0)
            return
        ####
        self._states = rng.normal(0.0, self._stds[:, None], size=self._states.shape)
    ####

    def step(self, dt: float, rng: random.Generator) -> ndarray:
        if dt <= 0 or self._taus.size == 0:
            return zeros(3)
        ####
        if dt != self._last_dt:
            self._last_phi, innovation_variance = flicker_transition_parameters(self._taus, self._stds, dt)
            self._last_innovation_std = sqrt(innovation_variance)
            self._last_dt = dt
        ####
        self._states = self._last_phi[:, None] * self._states + rng.normal(
            0.0,
            self._last_innovation_std[:, None],
            size=self._states.shape,
        )
        return sum(self._states, axis=0)
    ####
####
