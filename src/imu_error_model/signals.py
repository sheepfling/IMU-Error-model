from dataclasses import dataclass

from numpy import ndarray, zeros


@dataclass(slots=True, frozen=True)
class ImuOutput:
    """Sensor increments in the body frame over an explicit time interval.

    `delta_v` is the accelerometer velocity increment in m/s, expressed in
    body axes at `start_time`. `delta_theta` is the gyroscope rotation
    vector in radians from `start_time` to `end_time`, also expressed in
    body axes at `start_time`. `dt` is derived from the interval bounds;
    the rate properties are convenience views of these increments.
    """

    delta_v: ndarray
    delta_theta: ndarray
    start_time: float
    end_time: float
    temperature_celsius: float | None  # degrees Celsius (°C), or None when unavailable

    @property
    def dt(self) -> float:
        """Duration of the sampled interval in seconds."""
        return self.end_time - self.start_time
    ####

    @property
    def acceleration(self) -> ndarray:
        """Return the equivalent constant body-frame acceleration in m/s²."""
        return self.delta_v / self.dt if self.dt else zeros(3)
    ####

    @property
    def angular_rate(self) -> ndarray:
        """Return the equivalent constant body-frame angular rate in rad/s."""
        return self.delta_theta / self.dt if self.dt else zeros(3)
    ####
####
