from numpy import array, column_stack, ones
from numpy.linalg import lstsq

from playground.calibration.uncertainty import linear_fit_diagnostics


def test_linear_fit_diagnostics_identifies_full_rank_design() -> None:
    x = array((-2.0, -1.0, 0.0, 1.0, 2.0))
    design = column_stack((x, ones(x.size)))
    observations = column_stack((2.0 * x + 1.0, -0.5 * x + 3.0))
    coefficients, _, _, _ = lstsq(design, observations, rcond=None)

    diagnostics = linear_fit_diagnostics(design, observations, coefficients)

    assert diagnostics.full_rank
    assert diagnostics.rank == 2
    assert diagnostics.parameter_count == 2
    assert diagnostics.sample_count == 5
    assert diagnostics.condition_number < 2.0
    assert abs(diagnostics.residual_rms).max() < 1e-12
####


def test_linear_fit_diagnostics_reports_rank_deficiency() -> None:
    design = array(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)))
    observations = array(((1.0,), (2.0,), (3.0,)))
    coefficients, _, _, _ = lstsq(design, observations, rcond=None)

    diagnostics = linear_fit_diagnostics(design, observations, coefficients)

    assert not diagnostics.full_rank
    assert diagnostics.rank == 1
####
