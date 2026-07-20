from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImuOutput:
    """Sensor increments in the body frame at interval start.

    ``delta_v`` is in m/s, ``delta_theta`` is in radians, and ``dt`` is in
    seconds. The rate properties are convenience views of those increments.
    """

    delta_v: np.ndarray
    delta_theta: np.ndarray
    dt: float
    temperature: float

    @property
    def acceleration(self) -> np.ndarray:
        return self.delta_v / self.dt if self.dt else np.zeros(3)

    @property
    def angular_rate(self) -> np.ndarray:
        return self.delta_theta / self.dt if self.dt else np.zeros(3)
