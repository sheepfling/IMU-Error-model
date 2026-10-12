# Calibration recovery provenance

## Recovered inputs

The Tranche 9 source archive and companion development bundle were recovered on
2026-10-12 and checked locally:

- `imu-error-model-calibration-tranche9.zip` — SHA-256
  `a14fcd39df641d03f9347315567114b35a1864a223457af0435854e56ae797cb`.

- `imu-calibration-tranche9-development-bundle.zip` — SHA-256
  `77592575d21c235f0bd92be8fc2b9bd4df8865cd948dba08aaec0115886212e1`.

- `imu-calibration-tranche9-validation.zip` — SHA-256
  `fa013cf86025eaa77fa5d96b4874b661f06f6ea4c3db359f35cd954d744737e0`.

All three ZIPs passed CRC checks. The integration is based on `main` at
`df50d091c35960100ea29c3ec74c43697758f9e7`. The bundled T9 patch targets an
exact Tranche 8 snapshot and does not apply to that older `main`, so the
canonical current source snapshot was compared with `main` and integrated as a
source-level diff. The archives, patch, wheel, generated results, caches, and
nested bundles remain outside the repository.

The Tranche 6 Ferraris X/Y dynamic Earth-rate frame correction is retained with
its regression coverage in
`tests/test_reference_equipment.py::test_legacy_turntable_earth_projection_uses_correct_physical_axis`.

## Historical validation evidence

The companion Tranche 9 validation report recorded 347 collected tests, 346
passed, and one dependency-gated skip for the optional live `imucal` route.
That is historical bundle evidence; it is not a result of the local checks for
this integration. The report also marked Ruff and Pyright unavailable in its
validation environment.

## Local verification

Local checks on Python 3.13.15 completed on 2026-10-12:

- The full test suite passed with 346 passed and one optional `imucal` skip.
- The wheel build, isolated install, package-content check, and installed-package
  test run passed with the same 346 passed and one skipped result.

- Ruff, scope-marker validation, Markdown validation, and `pip check` passed.
- Pyright passed for `src`, `examples`, and `playground` with zero diagnostics.
- The separate Pyright test configuration reports 90 typing diagnostics in
  recovered tests. This remains an open quality gate; the draft is not
  release-ready.

- The Tranche 6 Ferraris X/Y frame regression passed as a focused test.

No hosted CI workflow was dispatched. The optional live `imucal` comparison
was not run because that extra is not installed in the local environment.

## Optional dependency status

The current source keeps `imucal` behind the `calibration` extra. PyPI's
project page lists 2.6.0 as the latest release and declares Python
`>=3.9,<3.14`, so the source constraint `imucal>=2.6,<3` includes the current
release and supports the project's Python 3.11–3.13 range. This metadata was
checked on 2026-10-12; it does not constitute a live optional-adapter test.
The release metadata was checked on the official PyPI project page.
