from pathlib import Path
from typing import Any

import pytest
import yaml

from imu_error_model import ImuConfig, LoadedProfile, load_profile, load_profile_document


@pytest.fixture(scope="session")
def standard_gravity() -> float:
    return 9.80665
####

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
####

@pytest.fixture(scope="session")
def hardware_profile_dir(project_root: Path) -> Path:
    return project_root / "examples" / "imu_profiles" / "hardware-estimates"
####

@pytest.fixture(scope="session")
def hardware_profile_paths(hardware_profile_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(hardware_profile_dir.glob("*.yaml")))
####

@pytest.fixture(scope="session")
def hardware_profile_texts(hardware_profile_paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in hardware_profile_paths
    }
####

@pytest.fixture(scope="session")
def hardware_profile_payloads(hardware_profile_paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in hardware_profile_paths
    }
####

@pytest.fixture(scope="session")
def hardware_profiles(hardware_profile_paths: tuple[Path, ...]) -> dict[str, LoadedProfile]:
    return {
        path.name: load_profile_document(path)
        for path in hardware_profile_paths
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
def hg9900_profile_path(hardware_profile_dir: Path) -> Path:
    return hardware_profile_dir / "hg9900.yaml"
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
