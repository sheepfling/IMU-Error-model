import numpy as np

from .config import AxisConfig


def white_noise(density: float, dt: float, rng: np.random.Generator, covariance: np.ndarray | None = None) -> np.ndarray:
    """Sample rate noise from a one-sided density, proportional to 1/sqrt(dt)."""
    if dt <= 0:
        return np.zeros(3)
    if covariance is not None:
        return rng.multivariate_normal(np.zeros(3), np.asarray(covariance) / dt)
    if density == 0:
        return np.zeros(3)
    return rng.normal(0.0, density / np.sqrt(dt), 3)


def gauss_markov(
    current: np.ndarray,
    std: float,
    correlation_time: float | None,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Advance a stationary first-order Gauss–Markov process."""
    if dt <= 0 or std == 0:
        return current.copy()
    if correlation_time is None:
        return current + rng.normal(0.0, std * np.sqrt(dt), 3)
    phi = np.exp(-dt / correlation_time)
    return phi * current + rng.normal(0.0, std * np.sqrt(1.0 - phi * phi), 3)


class FlickerBiasProcess:
    """Finite-band 1/f-like bias from logarithmically spaced OU processes.

    ``target_allan_std`` is calibrated at the geometric-center averaging time
    of the configured correlation-time band. This is an approximation intended
    for matching a finite datasheet Allan-deviation plateau.
    """

    def __init__(self, config: AxisConfig):
        self._target = config.flicker_bias_std
        if self._target == 0:
            self._taus = np.zeros(0)
            self._states = np.zeros((0, 3))
            self._stds = np.zeros(0)
            return
        assert config.flicker_min_correlation_time is not None
        assert config.flicker_max_correlation_time is not None
        self._taus = np.geomspace(
            config.flicker_min_correlation_time,
            config.flicker_max_correlation_time,
            config.flicker_components,
        )
        center = np.sqrt(self._taus[0] * self._taus[-1])
        unit_variances = np.array([self._allan_variance(1.0, tau, center) for tau in self._taus])
        scale = self._target / np.sqrt(np.sum(unit_variances) / config.flicker_components)
        self._stds = np.full(config.flicker_components, scale / np.sqrt(config.flicker_components))
        self._states = np.zeros((config.flicker_components, 3))

    @staticmethod
    def _allan_variance(variance: float, correlation_time: float, averaging_time: float) -> float:
        ratio = averaging_time / correlation_time
        exp_term = np.exp(-ratio)
        average_variance = 2.0 * variance * (correlation_time * averaging_time - correlation_time**2 * (1.0 - exp_term)) / averaging_time**2
        adjacent_covariance = variance * correlation_time**2 * (1.0 - exp_term) ** 2 / averaging_time**2
        return average_variance - adjacent_covariance

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            self._states.fill(0.0)
            return
        self._states = rng.normal(0.0, self._stds[:, None], size=self._states.shape)

    def step(self, dt: float, rng: np.random.Generator) -> np.ndarray:
        if dt <= 0 or self._taus.size == 0:
            return np.zeros(3)
        phi = np.exp(-dt / self._taus)
        innovation_std = self._stds * np.sqrt(1.0 - phi**2)
        self._states = phi[:, None] * self._states + rng.normal(0.0, innovation_std[:, None], size=self._states.shape)
        return np.sum(self._states, axis=0)
