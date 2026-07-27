from imu_error_model import (
    list_example_profile_categories,
    list_example_profiles,
    load_example_profile,
    load_noiseless_profile,
    read_example_profile,
)


def test_packaged_example_profiles_are_discoverable() -> None:
    names = list_example_profiles()
    assert len(names) == 14
    assert names == tuple(sorted(names))
    assert "ADIS16470.yaml" in names
    assert "hg9900.yaml" in names
####


def test_all_profile_categories_and_qualified_names_are_discoverable() -> None:
    assert list_example_profile_categories() == ("baselines", "hardware_estimates")
    names = list_example_profiles(None)
    assert len(names) == 15
    assert names == tuple(sorted(names))
    assert "baselines/ideal.yaml" in names
    assert "hardware_estimates/ADIS16470.yaml" in names
####


def test_packaged_example_profile_loads_with_logical_source() -> None:
    profile = load_example_profile("hg9900.yaml")
    assert profile.model_name == "HG9900"
    assert str(profile.source_path).startswith("package:")
    assert "notional" in read_example_profile("hg9900.yaml").lower()
####


def test_packaged_example_profile_resolution_accepts_paths_and_short_names() -> None:
    by_stem = load_example_profile("sbg_pulse_40")
    by_path = load_example_profile("hardware_estimates/sbg_pulse_40.yaml")
    by_path_with_category = load_example_profile(
        "hardware_estimates/sbg_pulse_40.yaml",
        category="hardware_estimates",
    )
    by_category = load_example_profile("ideal", category="baselines")
    assert by_stem.model_name == by_path.model_name == by_path_with_category.model_name == "SBG_PULSE_40"
    assert by_category.model_name == "Ideal noiseless IMU"
####


def test_packaged_example_profile_rejects_resource_traversal() -> None:
    try:
        load_example_profile("../hg9900.yaml")
    except ValueError as error:
        assert "invalid profile resource name" in str(error)
    else:
        raise AssertionError("resource traversal should be rejected")
####


def test_packaged_noiseless_profile_is_separate_from_hardware_estimates() -> None:
    assert list_example_profiles("baselines") == ("ideal.yaml",)
    profile = load_noiseless_profile()
    for config in (profile.config.accelerometer, profile.config.gyroscope):
        assert config.white_noise_density == 0.0
        assert config.turn_on_bias_std == 0.0
        assert config.bias_std == 0.0
        assert config.flicker_bias_std == 0.0
        assert config.scale_factor == 0.0
        assert config.nonlinear_factor == 0.0
        assert config.misalignment_std == 0.0
        assert config.quantization_step is None
    ####
####
