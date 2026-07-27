"""Composable stochastic and parametric error models for IMU increments."""

from .characterization import (
    ImuChannelErrorProcessEstimate,
    ImuErrorProcessEstimate,
    PersistentProcessEstimate,
    characterize_imu_error_process,
)
from .checkpoint import ImuModelCheckpoint
from .checkpoint_codecs import CheckpointCodecProtocol, PydanticJsonCheckpointCodec
from .config import DEFAULT_REFERENCE_TEMPERATURE_C, AxisConfig, ImuConfig, ProfileMetadata, ProfileSource
from .example_profiles import (
    list_example_profile_categories,
    list_example_profiles,
    load_example_profile,
    load_noiseless_profile,
    read_example_profile,
)
from .interfaces import (
    CheckpointableImuModelProtocol,
    ImuModelProtocol,
    SerializableCheckpointableImuModelProtocol,
)
from .model import ImuModel
from .profiles import (
    LoadedProfile,
    config_from_mapping,
    load_profile,
    load_profile_document,
    load_profile_document_stream,
    load_profile_stream,
    save_profile,
)
from .signals import ImuOutput
from .thermal import LinearThermalModel, ThermalModel, ThermalState


__all__ = [
    "AxisConfig", "ImuConfig", "ImuModel", "ImuOutput", "DEFAULT_REFERENCE_TEMPERATURE_C",
    "ImuModelProtocol", "CheckpointableImuModelProtocol", "ImuModelCheckpoint",
    "SerializableCheckpointableImuModelProtocol",
    "CheckpointCodecProtocol", "PydanticJsonCheckpointCodec",
    "ImuChannelErrorProcessEstimate", "ImuErrorProcessEstimate", "PersistentProcessEstimate",
    "characterize_imu_error_process",
    "LinearThermalModel", "ThermalModel", "ThermalState",
    "config_from_mapping", "load_profile", "load_profile_stream", "save_profile",
    "LoadedProfile", "load_profile_document", "load_profile_document_stream",
    "list_example_profiles", "load_example_profile", "load_noiseless_profile", "read_example_profile",
    "list_example_profile_categories",
    "ProfileMetadata", "ProfileSource",
]
