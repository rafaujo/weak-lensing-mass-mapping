# Reproduction notes and publication record

The new archive resolves the missing prior-construction script, MCALens test
driver, final GS reduced-shear outputs and validation fusion summaries. The
historical noise-calibration source has also been located and excerpted with
provenance. The seven original benchmark tables and the reduced-shear numbers pass the
archived-data consistency checks.

## Methodological and licensing provenance

- Fusion selection rationale recovered: see `provenance/fusion_selection/REPORT.txt`.
  Describe the qualitative validation choice accurately; do not claim a formal
  automatic objective or that no constituent-method test results had been seen.
- **Own-code license selected and applied:** MIT, copyright 2026 Rafael L. de Araujo.
  Preserve the component boundaries and third-party terms in `LICENSE_SCOPE.md`.
  The author keeps Joaquín's role in the existing acknowledgement; this does
  not relicense any retained third-party adaptation as original code.

## Execution and environment

- **Planned single-case smoke complete.** KS, Starlet, G+S and native GLIMPSE
  have independent v5 map checks. The v6 MCALens recheck confirms the packaged
  driver and new NPZ destination; the server worker records unchanged historical
  SHA256/size/mtime. Independent map metrics differ from the frozen CSV by at
  most 1.21e-9. See `provenance/server_validation_v6/`.
- No further server execution is required to close the MCALens isolation fix.
  All executable code, configuration, scientific inputs and results in this
  candidate are byte-identical to the tested v6 package.
- The smoke loader omits the three unused full-benchmark GS cube allocations.
  Their absence was covered by the existing offline regression test; v6's
  returned server run used MCALens-only and did not execute the GS loader.
- The actual imported gob5 pywt reports 1.8.0 while distribution metadata says
  1.9.0. Preserve the observed evidence; the dependency list is not a tested lock.
- A fresh-environment installation, full 480-case rerun, input-noise regeneration,
  bootstrap regeneration and figure regeneration are outside this smoke's scope.
  These limits remain documented; they do not reopen the completed one-case test.

## Selection provenance

- **MCALens threshold selection.** The supplied resumable validation driver is
  saved with `NSIGMAS=[6]`. The final-map validation CSV contains only 240 cases
  at Nsigma=5, and uses a different schema (`nsigma` instead of `Nsigma`). It
  cannot be used directly as that driver's resume file. The frozen map producer
  and test driver both use 5, consistent with the paper. Document the historical
  candidate thresholds and their validation scores if claiming to reproduce the
  threshold-selection procedure; do not infer the full grid from `[6]`. This is
  a limitation of the historical tuning record, not a missing input to the
  final frozen Nsigma=5 experiment. It need not block the final-code release.
- **Fusion weights: historical rationale recovered.** The project conversation
  explicitly weighs near-minimal NMSE against improved spectral/amplitude and
  peak diagnostics, before executing the frozen fusion test. No algorithm or
  weight changed. See the dated evidence and full metric comparison.
- The initial KS/Starlet fine-selection summaries describe full survey fields,
  not the final square patches. Preserve that distinction in the workflow.

## Frozen-input regeneration

- The exact prior array and its construction script are present. Regeneration
  against the archived array has not been tested.
- The notebook calibration excerpts use 64 noise draws, seeds 10000–10063 and
  the full survey-mask geometry. Both historical cells write the same CSV name;
  no outputs were supplied to identify which produced the frozen coefficients.
  Use the frozen coefficients for the existing benchmark; document or verify
  their regeneration before claiming an exact calibration rerun.

## Peak-zero analysis included in v1.1.0

The complete analysis and archived outputs are in `analysis/peak_zero/`.
The 480-case server analysis and original audit passed. The core benchmark
has not been rerun or retuned. See the current README and CHANGELOG for the
new package scope; dated v1.0.0 validation records below remain historical.

## Publication metadata

- License choice is complete for Rafael's own contributions. GLIMPSE and
  CosmoStat notices/licenses are preserved unchanged.
- Repository: `rafaujo/weak-lensing-mass-mapping`; release **v1.0.0** is archived
  at [10.5281/zenodo.23102124](https://doi.org/10.5281/zenodo.23102124). This is the historical v1.0.0 DOI;
  all 200 archived files match the release commit. Use the new v1.1.0 DOI for the revised manuscript;
  the peak-zero package was not present in v1.0.0.
- The completed Zenodo archive is separate from its downstream Software Heritage
  archival queue. The latter was still pending at the time of verification.

Only planned filesystem expressions and a configuration import were changed in
scientific sources. Algorithms, constants, frozen inputs and results are preserved.
