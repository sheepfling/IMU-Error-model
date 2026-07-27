"""Smoke tests for the package as installed from its built wheel."""

import os
from importlib.resources import files
from pathlib import Path

import imu_error_model
from imu_error_model import list_example_profiles, load_example_profile, load_noiseless_profile


def test_installed_wheel_exposes_all_packaged_profiles() -> None:
    """Verify the installed package exposes every profile resource and loader path."""
    expected_root = os.environ.get("IMU_ERROR_MODEL_EXPECTED_PACKAGE_ROOT")
    if expected_root is not None:
        package_path = imu_error_model.__file__
        assert package_path is not None
        package_file = Path(package_path).resolve()
        assert package_file.is_relative_to(Path(expected_root).resolve())
    ####
    assert files("imu_error_model.data.example_profiles").joinpath("catalog.json").is_file()
    profile_names = list_example_profiles(None)
    assert profile_names
    for name in profile_names:
        loaded = load_example_profile(name)
        assert loaded.config.accelerometer is not None
        assert loaded.config.gyroscope is not None
    ####
    assert load_noiseless_profile().metadata.family == "ideal"
####
