"""Generate compact tranche-8 continuous-acquisition evidence."""

from __future__ import annotations

import json
from pathlib import Path

from .continuous_support import alias_amplitude, fit_analog_gyro_time_constant, physical_clock_angle_error_ppm


def main() -> None:
    output = Path('artifacts/tranche8')
    output.mkdir(parents=True, exist_ok=True)
    tau = fit_analog_gyro_time_constant(
        frequencies_hz=(1.0, 2.0, 4.0, 7.0), sample_period_s=0.002, duration_s=4.0, analog_tau_s=0.025,
    )
    alias_raw = alias_amplitude(input_frequency_hz=80.0, sample_period_s=0.01, duration_s=4.0, analog_tau_s=0.0)
    alias_filtered = alias_amplitude(input_frequency_hz=80.0, sample_period_s=0.01, duration_s=4.0, analog_tau_s=0.01)
    payload = {
        'analog_tau_truth_s': 0.025,
        'analog_tau_estimate_s': tau.time_constant_s,
        'analog_tau_per_frequency_s': tau.per_frequency_s,
        'alias_input_hz': 80.0,
        'alias_folded_hz': alias_raw.frequency_hz,
        'alias_amplitude_no_analog_filter': alias_raw.amplitude,
        'alias_amplitude_tau_0p01_s': alias_filtered.amplitude,
        'alias_suppression_ratio': alias_filtered.amplitude / alias_raw.amplitude,
        'physical_clock_injected_ppm': 750.0,
        'nominal_count_angle_error_ppm': physical_clock_angle_error_ppm(
            frequency_error_ppm=750.0, sample_count=1000, nominal_period_s=0.01,
        ),
    }
    (output / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
####


if __name__ == '__main__':
    main()
####
