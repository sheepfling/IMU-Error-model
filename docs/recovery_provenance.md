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

Local checks on Python 3.13.15 completed on 2026-10-17:

- The full test suite passed with 346 passed and one optional `imucal` skip.
- The wheel build, isolated install, package-content check, and installed-package
  test run passed with the same 346 passed and one skipped result.

- Ruff, scope-marker validation, Markdown validation, and `pip check` passed.
- Pyright passed both configured profiles with zero diagnostics: source
  (`src`, `examples`, and `playground`) and the separate test profile.

- Before these type-only corrections, the hosted Python 3.11 source profile in
  failed Actions run `38286587470`
  reported one `reportReturnType` error in
  `playground/calibration/reference_identification.py`. That run stopped before
  reaching the test profile. A local baseline of the test profile reported 90
  diagnostics (88 `reportArgumentType`, 2 `reportOperatorIssue`). After the
  corrections, both local profiles report zero diagnostics.

- The Tranche 6 Ferraris X/Y frame regression passed as a focused test.

The optional live `imucal` comparison was not run because that extra is not
installed in the local environment. No workflow was dispatched by request; the
PR's normal pull-request workflow is the hosted validation path.

## Optional dependency status

The current source keeps `imucal` behind the `calibration` extra. As checked on
2026-10-17, the PyPI project page lists 2.6.0 as its latest release, uploaded
on 2024-10-31, with `Requires-Python >=3.9,<3.14` and Python 3.9–3.13
classifiers. The source constraint `imucal>=2.6,<3` includes that release and
its declared interpreter range covers this project's Python 3.11 and 3.12
support. Metadata compatibility is not a live optional-adapter test; the extra
was not installed or exercised here.
