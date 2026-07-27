from numpy import deg2rad, isclose

from imu_error_model import LoadedProfile


def load_example(name: str, hardware_profiles: dict[str, LoadedProfile]) -> LoadedProfile:
    return hardware_profiles[name]
####

def test_all_hardware_examples_load_with_provenance(
        hardware_profile_paths: tuple[str, ...],
        hardware_profiles: dict[str, LoadedProfile],
        hardware_profile_texts: dict[str, str],
) -> None:
    assert len(hardware_profile_paths) >= 10
    for name in hardware_profile_paths:
        profile = hardware_profiles[name]
        assert profile.model_name
        assert profile.sample_period_s > 0
        assert profile.model_name in hardware_profile_texts[name]
        assert profile.config.accelerometer.measurement_range is not None
        assert profile.config.gyroscope.measurement_range is not None
        assert set(profile.metadata.model_dump()) == {
            "family",
            "vendor",
            "grade",
            "revision",
            "source_date",
            "sources",
            "tags",
            "active",
        }
        sources = profile.metadata.sources
        assert sources
        assert all(source.url.scheme in {"http", "https"} for source in sources)
        assert all(len(source.date.isoformat()) == 10 for source in sources)
        assert all(source.archive_urls is not None for source in sources)
        assert all(
            archive_url.scheme in {"http", "https"}
            for source in sources
            for archive_url in source.archive_urls
        )
    ####
####


def test_all_hardware_examples_are_explicitly_notional(
        hardware_profile_paths: tuple[str, ...],
        hardware_profile_texts: dict[str, str],
) -> None:
    for name in hardware_profile_paths:
        text = hardware_profile_texts[name].lower()
        assert "notional" in text
        assert "approximate modeling estimates" in text
        assert "not official" in text
    ####
####


def test_hg1700_ag58_uses_product_sheet_range_rate_and_noise(
        hardware_profiles: dict[str, LoadedProfile],
        standard_gravity: float,
) -> None:
    profile = load_example("hg1700ag58.yaml", hardware_profiles)
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert isclose(profile.sample_period_s, 0.01)
    accel_range = accel.measurement_range
    gyro_range = gyro.measurement_range
    assert accel_range is not None
    assert gyro_range is not None
    assert isclose(accel_range, 37.0 * standard_gravity)
    assert isclose(accel.white_noise_density, 0.02 / 60.0)
    assert isclose(gyro_range, deg2rad(1074.0))
    assert isclose(gyro.white_noise_density, deg2rad(0.125) / 60.0)
####


def test_hg1700_ag71_uses_reference_performance_table(
        hardware_profiles: dict[str, LoadedProfile],
        standard_gravity: float,
) -> None:
    profile = load_example("hg1700ag71.yaml", hardware_profiles)
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert isclose(profile.sample_period_s, 0.01)
    accel_range = accel.measurement_range
    gyro_range = gyro.measurement_range
    assert accel_range is not None
    assert gyro_range is not None
    assert isclose(accel_range, 70.0 * standard_gravity)
    assert isclose(accel.scale_factor, 300e-6)
    assert isclose(accel.nonlinear_factor, 500e-6)
    assert isclose(accel.white_noise_density, 0.065 * 0.3048 / 60.0)
    assert isclose(gyro_range, deg2rad(1017.0))
    assert isclose(gyro.scale_factor, 150e-6)
    assert isclose(gyro.nonlinear_factor, 150e-6)
    assert isclose(gyro.white_noise_density, deg2rad(0.125) / 60.0)
####


def test_hg5700_profiles_include_published_scale_factors(
        hardware_profiles: dict[str, LoadedProfile],
) -> None:
    for name in ("hg5700ca01.yaml", "hg5700ba01.yaml", "hg5700aa01.yaml"):
        profile = load_example(name, hardware_profiles)
        assert isclose(profile.config.accelerometer.scale_factor, 120e-6)
        assert isclose(profile.config.gyroscope.scale_factor, 40e-6)
        assert isclose(profile.config.accelerometer.white_noise_density, 0.065 * 0.3048 / 60.0)
    ####
####


def test_hg5700_ca01_prefers_detailed_brochure_arw(
        hardware_profiles: dict[str, LoadedProfile],
) -> None:
    profile = load_example("hg5700ca01.yaml", hardware_profiles)
    assert isclose(profile.config.gyroscope.white_noise_density, deg2rad(0.0062) / 60.0)
####


def test_adis_profile_exposes_explicit_allan_curve_choices(
        hardware_profiles: dict[str, LoadedProfile],
        standard_gravity: float,
) -> None:
    profile = load_example("ADIS16470.yaml", hardware_profiles)
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert isclose(accel.white_noise_density, 0.037 / 60.0)
    assert isclose(gyro.white_noise_density, deg2rad(0.34) / 60.0)
    assert accel.flicker_components == gyro.flicker_components == 16
    assert accel.flicker_min_correlation_time is not None
    assert gyro.flicker_min_correlation_time is not None
    assert accel.flicker_max_correlation_time is not None
    assert gyro.flicker_max_correlation_time is not None
    assert isclose(accel.flicker_min_correlation_time, 1.0)
    assert isclose(gyro.flicker_min_correlation_time, 1.0)
    assert isclose(accel.flicker_max_correlation_time, 1000.0)
    assert isclose(gyro.flicker_max_correlation_time, 1000.0)
    assert isclose(accel.flicker_bias_std, 8e-6 * standard_gravity)
    assert isclose(gyro.flicker_bias_std, deg2rad(8.0) / 3600.0)
####


def test_hg9900_uses_product_function_specification(
        hardware_profiles: dict[str, LoadedProfile],
        standard_gravity: float,
) -> None:
    profile = load_example("hg9900.yaml", hardware_profiles)
    accel = profile.config.accelerometer
    gyro = profile.config.gyroscope

    assert isclose(profile.sample_period_s, 1.0 / 300.0)
    assert isclose(accel.scale_factor, 117e-6)
    assert isclose(accel.white_noise_density, 0.0025 * 0.3048 / 60.0)
    assert isclose(accel.turn_on_bias_std, 25e-6 * standard_gravity)
    assert isclose(accel.bias_std, 100e-6 * standard_gravity)
    assert isclose(gyro.scale_factor, 5e-6)
    assert isclose(gyro.white_noise_density, deg2rad(0.0021) / 60.0)
    assert isclose(gyro.turn_on_bias_std, deg2rad(0.0041) / 3600.0)
####
