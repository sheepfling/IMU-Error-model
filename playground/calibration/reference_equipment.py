"""Lab-fixed rate-table kinematics; reference physics stays outside the IMU core.

NED is Earth fixed. R maps sensor to NED. Accelerometers measure specific force,
not apparent gravity: a stationary sensor sees -R.T @ g_ned. This module uses
co-moving channel averages, not navigation preintegrated rotation vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .earth_reference import LocalEarthReference


FloatArray = NDArray[np.float64]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector3 = tuple[Finite, Finite, Finite]


def readonly(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    """Own an immutable finite array with an exact shape."""
    result = np.asarray(value, dtype=float).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    ####
    result.setflags(write=False)
    return result
####


def skew(vector: FloatArray) -> FloatArray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
####


def rotation_vector(vector: Vector3) -> FloatArray:
    """Rodrigues map with stable sinc coefficients, including zero angle."""
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("rotation vector must be finite with shape (3,)")
    ####
    angle = float(np.linalg.norm(v))
    k = skew(v)
    return np.eye(3) + np.sinc(angle / np.pi) * k + 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * (k @ k)
####


class AngleLaw(BaseModel):
    """Smooth angle in radians as a linear drift plus a declared harmonic basis.

    Coefficients can come from commands OR encoder observations. Their source is
    owned by the calling boundary. Negative time is valid for clock alignment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    offset_rad: Finite = 0.0
    rate_radps: Finite = 0.0
    sine_rad: Finite = 0.0
    cosine_rad: Finite = 0.0
    frequency_hz: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 0.2

    def evaluate(self, times_s: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        t = np.asarray(times_s, dtype=float)
        if not np.isfinite(t).all():
            raise ValueError("times must be finite")
        ####
        nu = 2 * np.pi * self.frequency_hz
        sine, cosine = np.sin(nu * t), np.cos(nu * t)
        q = self.offset_rad + self.rate_radps * t + self.sine_rad * sine + self.cosine_rad * cosine
        w = self.rate_radps + nu * (self.sine_rad * cosine - self.cosine_rad * sine)
        alpha = -nu ** 2 * (self.sine_rad * sine + self.cosine_rad * cosine)
        return q, w, alpha
    ####

    def mean_rate(self, starts_s: FloatArray, ends_s: FloatArray) -> FloatArray:
        starts, ends = validate_intervals(starts_s, ends_s)
        return (self.evaluate(ends)[0] - self.evaluate(starts)[0]) / (ends - starts)
    ####
####


class TableGeometry(BaseModel):
    """One lab setup, or a declared surveyed estimate of that setup.

    R = R_tilt @ R_base @ Rx(wx(q)) @ Ry(wy(q)) @ Rz(q) @ R_mount.
    Wobble is angle-synchronous, not white attitude noise. The pivot is fixed in
    local NED; the lever arm runs from pivot to one colocated IMU triad center,
    expressed in sensor coordinates. Re-indexing/handling is a separate segment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    base_rotvec_ned_rad: Vector3 = (0.0, 0.0, 0.0)
    tilt_rotvec_ned_rad: Vector3 = (0.0, 0.0, 0.0)
    platform_from_sensor_rotvec_rad: Vector3 = (0.0, 0.0, 0.0)
    wobble_x_rad: Finite = 0.0
    wobble_y_rad: Finite = 0.0
    wobble_order: Annotated[int, Field(ge=1, le=32, strict=True)] = 1
    wobble_phase_rad: Finite = 0.0
    lever_arm_sensor_m: Vector3 = (0.0, 0.0, 0.0)
####


@dataclass(frozen=True, slots=True)
class KinematicSamples:
    """Evaluator/reference-provider values at physical times; not IMU output."""

    orientation_ned_from_sensor: FloatArray
    orientation_dot: FloatArray
    orientation_ddot: FloatArray
    specific_force_sensor: FloatArray
    inertial_rate_sensor: FloatArray
    relative_rate_sensor: FloatArray
    relative_acceleration_sensor: FloatArray
    lever_operator: FloatArray
####


def _axis_jet(axis: int, angle: FloatArray, rate: FloatArray, acceleration: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    k = skew(np.eye(3)[axis])
    r = np.eye(3) + np.sin(angle)[..., None, None] * k + (1 - np.cos(angle))[..., None, None] * (k @ k)
    dr = rate[..., None, None] * (r @ k)
    ddr = acceleration[..., None, None] * (r @ k) + rate[..., None, None] ** 2 * (r @ k @ k)
    return r, dr, ddr
####


def _product_jet(a: tuple[FloatArray, FloatArray, FloatArray], b: tuple[FloatArray, FloatArray, FloatArray]) -> tuple[FloatArray, FloatArray, FloatArray]:
    r, dr, ddr = a
    s, ds, dds = b
    return r @ s, dr @ s + r @ ds, ddr @ s + 2 * dr @ ds + r @ dds
####


def _vee(matrix: FloatArray) -> FloatArray:
    # Antisymmetrization rejects roundoff in theoretically skew matrices.
    return 0.5 * np.stack((matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 1, 0] - matrix[..., 0, 1]), axis=-1)
####


class ReferenceProvider(Protocol):
    def sample(self, times_s: FloatArray) -> KinematicSamples:
        ...
####


@dataclass(frozen=True, slots=True)
class RateTableReference:
    """Analytic orientation and derivatives for a compact lab-fixed trajectory.

    Normal *effective* gravity is uniform across the small table. Coriolis from
    lever motion is retained. Gravity gradients, transport, Earth-orientation
    variations, pivot translation and flexible bodies are deliberately absent.
    """

    earth: LocalEarthReference
    law: AngleLaw
    geometry: TableGeometry
    include_earth_rotation: bool = True

    def sample(self, times_s: FloatArray) -> KinematicSamples:
        t = np.asarray(times_s, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all():
            raise ValueError("query times must be a finite one-dimensional array")
        ####
        q, w, alpha = self.law.evaluate(t)
        g = self.geometry
        h = g.wobble_order
        phase = h * q + g.wobble_phase_rad
        wx, wy = g.wobble_x_rad * np.sin(phase), g.wobble_y_rad * np.cos(phase)
        dx, dy = g.wobble_x_rad * h * np.cos(phase) * w, -g.wobble_y_rad * h * np.sin(phase) * w
        ddx = g.wobble_x_rad * (h * np.cos(phase) * alpha - h ** 2 * np.sin(phase) * w ** 2)
        ddy = -g.wobble_y_rad * (h * np.sin(phase) * alpha + h ** 2 * np.cos(phase) * w ** 2)
        moving = _product_jet(_product_jet(_axis_jet(0, wx, dx, ddx), _axis_jet(1, wy, dy, ddy)), _axis_jet(2, q, w, alpha))
        left = rotation_vector(g.tilt_rotvec_ned_rad) @ rotation_vector(g.base_rotvec_ned_rad)
        right = rotation_vector(g.platform_from_sensor_rotvec_rad)
        r, dr, ddr = (left @ x @ right for x in moving)
        rt = np.swapaxes(r, -1, -2)
        omega = self.earth.earth_rotation_ned_radps if self.include_earth_rotation else np.zeros(3)
        relative_rate = _vee(rt @ dr)
        relative_acceleration = _vee(rt @ ddr)
        lever_operator = rt @ (ddr + 2 * skew(omega) @ dr)
        gravity = self.earth.gravity_ned_mps2
        f = -(rt @ gravity) + lever_operator @ np.asarray(g.lever_arm_sensor_m)
        inertial_rate = relative_rate + rt @ omega
        arrays = (r, dr, ddr, f, inertial_rate, relative_rate, relative_acceleration, lever_operator)
        for a in arrays:
            a.setflags(write=False)
        ####
        return KinematicSamples(*arrays)
    ####
####


def validate_intervals(starts_s: FloatArray, ends_s: FloatArray) -> tuple[FloatArray, FloatArray]:
    starts, ends = np.asarray(starts_s, dtype=float), np.asarray(ends_s, dtype=float)
    if starts.ndim != 1 or starts.size == 0 or starts.shape != ends.shape:
        raise ValueError("intervals must be nonempty one-dimensional arrays of equal shape")
    ####
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(ends <= starts):
        raise ValueError("intervals must have finite increasing endpoints")
    ####
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("intervals must be ordered and nonoverlapping")
    ####
    return starts, ends
####


@lru_cache(maxsize=16)
def _quadrature(order: int) -> tuple[FloatArray, FloatArray]:
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("quadrature order must be an integer in [2, 64]")
    ####
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights
####


@dataclass(frozen=True, slots=True)
class ApertureReference:
    """Aperture-mean sensor-channel components, with explicit physical intervals."""

    starts_s: FloatArray
    ends_s: FloatArray
    force: FloatArray
    rate: FloatArray
    lever_operator: FloatArray

    def __post_init__(self) -> None:
        starts, ends = validate_intervals(self.starts_s, self.ends_s)
        n = len(starts)
        for name, value, shape in (("starts_s", starts, (n,)), ("ends_s", ends, (n,)), ("force", self.force, (n, 3)), ("rate", self.rate, (n, 3)), ("lever_operator", self.lever_operator, (n, 3, 3))):
            object.__setattr__(self, name, readonly(value, shape, name))
        ####
    ####
####


def aperture_reference(provider: ReferenceProvider, starts_s: FloatArray, ends_s: FloatArray, *, order: int = 8) -> ApertureReference:
    """Gauss-Legendre channel integration; certify convergence for each campaign.

    This integrates changing body-axis components. It does not compute a coning-
    corrected SO(3) increment or sculling-corrected delta velocity. No claim of
    exactness is made for general wobble or nonconstant rates.
    """
    starts, ends = validate_intervals(starts_s, ends_s)
    nodes, weights = _quadrature(order)
    query = (starts[:, None] + ends[:, None]) / 2 + (ends - starts)[:, None] * nodes / 2
    samples = provider.sample(query.ravel())
    n = len(starts)
    f = np.einsum("q,nqj->nj", weights / 2, samples.specific_force_sensor.reshape(n, order, 3))
    w = np.einsum("q,nqj->nj", weights / 2, samples.inertial_rate_sensor.reshape(n, order, 3))
    lever = np.einsum("q,nqij->nij", weights / 2, samples.lever_operator.reshape(n, order, 3, 3))
    return ApertureReference(starts, ends, f, w, lever)
####
