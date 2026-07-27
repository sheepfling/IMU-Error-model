"""Composable stochastic and parametric error models for IMU increments."""

from .characterization import (
    ImuChannelErrorProcessEstimate,
    ImuErrorProcessEstimate,
    PersistentProcessEstimate,
    characterize_imu_error_process,
)
from .config import DEFAULT_REFERENCE_TEMPERATURE_C, AxisConfig, ImuConfig, ProfileMetadata, ProfileSource
from .interfaces import ImuModelProtocol
from .model import ImuModel
from .profiles import LoadedProfile, config_from_mapping, load_profile, load_profile_document, load_profile_stream, \
    save_profile
from .signals import ImuOutput
from .thermal import LinearThermalModel, ThermalModel, ThermalState


__all__ = [
    "AxisConfig", "ImuConfig", "ImuModel", "ImuOutput", "DEFAULT_REFERENCE_TEMPERATURE_C",
    "ImuModelProtocol",
    "ImuChannelErrorProcessEstimate", "ImuErrorProcessEstimate", "PersistentProcessEstimate",
    "characterize_imu_error_process",
    "LinearThermalModel", "ThermalModel", "ThermalState",
    "config_from_mapping", "load_profile", "load_profile_stream", "save_profile",
    "LoadedProfile", "load_profile_document",
    "ProfileMetadata", "ProfileSource",
]
