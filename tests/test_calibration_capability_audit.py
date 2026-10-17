from playground.calibration.capability_audit import (
    AXIS_EXCITATION_COVERAGE,
    CALIBRATION_ROUTE_COVERAGE,
    IMU_EXCITATION_COVERAGE,
    routes_without_executable_coverage,
    unclassified_config_fields,
)


def test_every_public_config_field_has_an_excitation_decision() -> None:
    assert unclassified_config_fields() == ()
    assert any(item.field_name == "nonlinear_factor" and item.status == "covered" for item in AXIS_EXCITATION_COVERAGE)
    assert any(
        item.field_name == "gyroscope_specific_force_sensitivity" and item.status == "covered"
        for item in IMU_EXCITATION_COVERAGE
    )
####


def test_named_calibration_routes_have_executable_or_diagnostic_evidence() -> None:
    assert routes_without_executable_coverage() == ()
    assert any(
        route.route_name == "imucal_ferraris_turntable" and route.status == "covered"
        for route in CALIBRATION_ROUTE_COVERAGE
    )
    assert any(
        route.route_name == "joint_data_budget_planning" and route.status == "covered"
        for route in CALIBRATION_ROUTE_COVERAGE
    )
    assert any(
        route.route_name == "high_rate_inertial_preintegration" and route.status == "covered"
        for route in CALIBRATION_ROUTE_COVERAGE
    )
####
