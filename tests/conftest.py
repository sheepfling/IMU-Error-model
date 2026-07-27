from pathlib import Path
from typing import Any

import pytest
import yaml

from imu_error_model import (
    ImuConfig,
    LoadedProfile,
    list_example_profiles,
    load_example_profile,
    load_profile,
    read_example_profile,
)


@pytest.fixture(scope="session")
def standard_gravity() -> float:
    return 9.80665
####

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
####

@pytest.fixture(scope="session")
def hardware_profile_paths() -> tuple[str, ...]:
    return list_example_profiles()
####

@pytest.fixture(scope="session")
def hardware_profile_texts(hardware_profile_paths: tuple[str, ...]) -> dict[str, str]:
    return {
        name: read_example_profile(name)
        for name in hardware_profile_paths
    }
####

@pytest.fixture(scope="session")
def hardware_profile_payloads(hardware_profile_paths: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        name: yaml.safe_load(read_example_profile(name))
        for name in hardware_profile_paths
    }
####

@pytest.fixture(scope="session")
def hardware_profiles(hardware_profile_paths: tuple[str, ...]) -> dict[str, LoadedProfile]:
    return {
        name: load_example_profile(name)
        for name in hardware_profile_paths
    }
####

@pytest.fixture(scope="session")
def test_profile_dir(project_root: Path) -> Path:
    return project_root / "tests" / "profiles"
####

@pytest.fixture(scope="session")
def test_profile_paths(test_profile_dir: Path) -> dict[str, Path]:
    return {
        path.relative_to(test_profile_dir).as_posix(): path
        for path in sorted(test_profile_dir.rglob("*.json"))
    }
####

@pytest.fixture(scope="session")
def test_profiles(test_profile_paths: dict[str, Path]) -> dict[str, ImuConfig]:
    return {
        name: load_profile(path)
        for name, path in test_profile_paths.items()
    }
####

@pytest.fixture(scope="session")
def hg9900_profile(hardware_profiles: dict[str, LoadedProfile]) -> LoadedProfile:
    return hardware_profiles["hg9900.yaml"]
####

@pytest.fixture(scope="session")
def sbg_pulse_40_profile(hardware_profiles: dict[str, LoadedProfile]) -> LoadedProfile:
    return hardware_profiles["sbg_pulse_40.yaml"]
####

@pytest.fixture(scope="session")
def short_correlation_profile(test_profiles: dict[str, ImuConfig]) -> ImuConfig:
    return test_profiles["test/short-correlation.json"]
####

@pytest.fixture(scope="session")
def flicker_band_profile(test_profiles: dict[str, ImuConfig]) -> ImuConfig:
    return test_profiles["test/flicker-band.json"]
####

@pytest.fixture(scope="session")
def short_correlation_profile_path(test_profile_paths: dict[str, Path]) -> Path:
    return test_profile_paths["test/short-correlation.json"]
####

@pytest.fixture(scope="session")
def flicker_band_profile_path(test_profile_paths: dict[str, Path]) -> Path:
    return test_profile_paths["test/flicker-band.json"]
####
