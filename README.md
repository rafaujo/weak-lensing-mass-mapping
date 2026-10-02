# Weak-lensing mass-mapping reproducibility code

Code and compact numerical results for **Sparse and Gaussian Priors for
Weak-Lensing Mass Mapping: Observable-Dependent Performance on FAIR Universe
Simulations**, by Rafael L. de Araujo and Reynam da C. Pestana, Federal University
of Bahia.

Repository: [rafaujo/weak-lensing-mass-mapping](https://github.com/rafaujo/weak-lensing-mass-mapping).
An archival release DOI has not yet been assigned.

**Status: the planned single-case server smoke test is complete.** The v5
artifacts verified KS, masked Starlet, G+S and native GLIMPSE for C0/R10/P0.
The v6 MCALens-only recheck verifies the packaged driver, a new output NPZ
inside the smoke directory, and unchanged historical-file hash/size/mtime
recorded by the server worker. Independent recalculation from the returned map
matches the archived metrics to within 1.21e-9. See
`provenance/server_validation_v6/ASSESSMENT.txt` and `independent_audit.json`.

This repository contains the tested v6 code unchanged, with updated documentation
and compact evidence. No further server check is needed for the isolation fix.
The smoke covers one case, not regeneration of all 480 patches or a fresh
environment installation. Rafael's own contributions are licensed under MIT;
third-party components retain their licenses. See [license scope](LICENSE_SCOPE.md).
A versioned release and Zenodo DOI remain to be prepared.

## Single-case server smoke test

See [README_SERVIDOR.txt](README_SERVIDOR.txt). The supplied server configuration
uses the recorded Linux paths and the existing scientific Python environment.
From this repository's root:

```bash
bash run_server_smoke.sh --preflight-only
bash run_server_smoke.sh
# Optional native GLIMPSE run (the v5 server smoke took about 14 minutes):
bash run_server_smoke.sh --with-glimpse
```

The runner executes C0/R10/P0 with the production KS, masked Starlet,
Gaussian+Starlet and MCALens functions. Each invocation creates a new output
directory; archived results and historical server caches are not overwritten.
The original iteration budgets remain intact. Comparisons against archived
case metrics are diagnostics, not a full 480-case reproduction. The v4 attempt failed on the server with missing `modopt`; v5 corrects the
interpreter selection. The v5 server results subsequently passed numerical artifact checks, but exposed
an MCALens output isolation defect. V6 passed 27 offline support tests and the
short MCALens-only server recheck. The code has not changed since those checks.

The development conversation used **two distinct Python environments**:
`/tatu/venv/gob5/bin/python` for the core pipeline, and
`~/venvs/mcalens/bin/python` for MCALens. The latter was built over the existing
micromamba `glimpse` environment with system-site-packages enabled. The v5
runner launches it through `micromamba run -p ~/micromamba/envs/glimpse` and
does not install MCALens dependencies into gob5. The preflight now checks real
imports, including modopt/seaborn/mass_mapping in the MCALens process.
`config_environments.server.toml` controls these commands separately from paths.
For full drivers, `bash run_server_pipeline.sh --dry-run run_mcalens_test.py`
shows the selected interpreter without executing the benchmark. See
`provenance/environments/historical_mapping.json` for the recovered evidence.

## Fusion selection provenance

The historical development conversation has now been recovered. It records a
qualitative choice based on validation field, spectral and peak diagnostics,
then a frozen test script and its outputs. No scalar multicriteria objective
was specified. The (.75,.75) point accepts a 0.0636% NMSE increase relative to
(.75,.50) for improvements in other observables; see
`provenance/fusion_selection/REPORT.txt` and `historical_decision.json`.
The supported timing claim is before the fusion's test evaluation. Constituent
method test results had already been seen during development. This does not
establish that the entire test set was untouched throughout development.

## Quick check of the manuscript tables

From this repository's root, with Python 3.12:

```bash
python tools/check_paper_tables.py
```

The checker uses only the Python standard library. It compares stored values and
case-level aggregates with the printed precision of all seven manuscript tables:
field metrics, approximate factorial comparison, spectra, peak statistics, peak
bootstrap intervals, photo-z trends, and the ADMM solver audit. The updated checker
passes 162 checks, including the final reduced-shear case grid, aggregates and
stored intervals. The exact checks are in
`provenance/paper_table_checks.json`. Expected printed numbers come from the
supplied manuscript; its hash is in `provenance/manuscript_reference.json`.

This is a consistency check of the supplied evidence. It does not rerun the
reconstruction solvers, recompute bootstrap samples, or validate all statements
in the article. This table checker does not regenerate bootstrap samples or reconstruction maps;
the separate one-case server smoke is documented above.

## Repository contents

```text
code/pipeline/       Core operator; Starlet, Gaussian and combined methods;
                     fusion; ADMM; robustness; bootstrap; final V12 figures
code/glimpse/        Final GLIMPSE validation/test drivers and FITS audit
config/pipeline/     Frozen split, KS/Starlet parameters and noise weights
config/glimpse/      The final lambda = 2.5, 3.0, 3.5 validation configurations
config/priors/       Exact validation-derived P_kappa array and its metadata
external/glimpse/   Recorded GLIMPSE source/build snapshot and upstream license
results/validation/ Selection evidence, validation results and ADMM audit
results/test/       Square-benchmark results and final comparisons
results/wiener/     Gaussian-only validation grid and test cases
results/paper/      Final V12 CSVs used by the manuscript figures and tables
tools/              Lightweight manuscript-table consistency check
provenance/         Checksums, source versions, patches, selection and paper map
```

The candidate retains the same 118 original files selected in v2. Of these,
94 remain byte-identical and 24 scientific scripts have only the documented path
expressions and configuration import changed. There are 25 scientific Python
scripts plus the new path helper. Another 31 supplied
files are duplicates of retained content; 166 are kept only in the local review
archive. The 17 manuscript files were reference material, not files to redistribute
as part of the code release. Generated repository documentation and checking tools
are additional files. No algorithm, numerical result, hyperparameter or random
seed has changed. `provenance/curation_manifest.csv` preserves the v2 selection
and original checksums. `provenance/portability/planned_changes.json` and
`transformed_files.json` record the subsequent path-only transformation.

Two calibration excerpts from notebook cells 2 and 3 (zero-based indices) are
stored as text in `provenance/noise_calibration/`, with exact source ranges and
hashes. They document 64 noise draws and seeds starting at 10000, using the full
survey mask. They are historical source evidence, not supported standalone
runners. The notebook arrived without outputs, so the cell that produced the
frozen coefficients and their numerical regeneration have not been verified.

Only V12 is the final figure generator. The supplied `fig_ablation_revised.pdf`
is not referenced by the manuscript; the ablation is represented by tables.
Earlier peak/spectral diagnostics use different definitions and must not be
substituted for the V12 results. Older full-field `fair_TEST_*` tables are likewise
not the final 480-patch benchmark.

The `run_reweighted_gaussian_starlet_validation.py` script is retained because
its first stage creates the validation GS reconstruction and component maps used
by fusion and Appendix A. Its later exploratory reweighting stage is not a
reported final method. It has been left intact; a future stage-only entry point
should be checked against the saved first-stage outputs.

## Raw data and frozen inputs

Obtain these files from the
[FAIR Universe public release, v1](https://zenodo.org/records/20056065):

| File | Shape | MD5 |
| --- | --- | --- |
| `sampled_WIDE12H_bin2_2arcmin_kappa.npy` | `(3, 30, 132019)` | `63388a13ed245208a05a4f66c8f188c1` |
| `sampled_label.npy` | `(3, 30, 5)` | `9a08befb4c635966cf61ced1136e7144` |
| `WIDE12H_bin2_2arcmin_mask.npy` | `(1424, 176)` | `29b8bbb8a9ad41d3dacfaf61008a8268` |

These three files total 24,018,028 bytes. The checksums collected on the authors'
original server match the published sample checksums. Both MD5 and SHA256 are
recorded in `provenance/raw_data_checksums.json`. The full 6.8 GB convergence array
is not required by the audited scripts; raw data are not redistributed here.

The shared validation prior is included at
`config/priors/fair_validation_pk_mean.npy`: 1,112 bytes, 123 finite nonnegative
float64 values. Its metadata describes aggregation of 240 validation patches and
excludes test truth. Its recovered construction script is
`code/pipeline/build_fair_validation_pk.py`; it reads the validation observation
cache and estimates the mean spectrum over 240 patches. It has not been rerun
or numerically compared with the frozen array. Existing consumers can use that
exact archived array directly.

## License and credits

Rafael L. de Araujo's original contributions are available under the
[MIT license](LICENSE), copyright 2026 Rafael L. de Araujo. GLIMPSE retains
CeCILL 2.1, and CosmoStat/MCALens retains its MIT license and CosmoStat notices.
Read [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for the component boundaries and
third-party attribution. The root license does not relicense external code.

The author's MIT approval is recorded in `provenance/licensing/author_decision.json`.
The existing acknowledgement to Joaquín A. Acedo for the initial ADMM contribution
is preserved; the paper's citation authors are unchanged.

The latest documentation/license-only changes and preserved file hashes are in
`provenance/licensing/package_changes.json`. Earlier audit manifests describe
their dated preparation snapshots and remain as historical evidence.

## Environment and upstream software

The gob5 environment recorded Python 3.12.3 on Linux. `requirements.txt` lists
its dependencies only; it must not be applied as a shared MCALens lock. It is an
initial dependency list, not an installation-tested environment lock.

The original PyTorch build was a local NVIDIA wheel,
`torch-2.9.0a0+145a3a7bda.nv25.10-36764868-cp312-cp312-linux_x86_64.whl`.
Its path/hash is preserved in `provenance/python_environment_original.txt`.
An ordinary `pip install torch` is not evidence of recreating this build. Recover
the original container identifier or wheel, or validate an alternative environment.

MCALens came from `https://github.com/CosmoStat/cosmostat.git`, commit
`7ae457db1697ac966fc6e739c00b91fa6a47ec2c`, **with a local modification** to
`pycs/sparsity/sparse2d/starlet.py`. The recorded patch is in
`provenance/mcalens/local_source_changes.patch`. It guards the optional compiled
`MRStarlet` class behind `PYSAP_CXX` and sets `MRStarlet = None` when those bindings
are unavailable. The full diff also records line-ending changes; a readable
normalized explanation is provided separately. Merely checking out the commit
omits that local modification. Import the corresponding `pycs` checkout, not an
unrelated package with the same name. The MIT license at that exact upstream
commit was retrieved and is preserved beside the patch in
`provenance/mcalens/LICENSE` (Git blob `2d32ea5d8f38c6c22dfada6f87c1183866d887a6`).
The newly supplied `starlet.py` matches the result encoded by that patch after
normalizing line endings, so another copy of the upstream library is unnecessary.

GLIMPSE came from `https://github.com/CosmoStat/Glimpse.git`, commit
`a0e09a4a3faf050e4809fc537d6b45a626ece4fc`. The included snapshot already contains
the two recorded changes in CMake and `src/survey.cpp`; do not apply its provenance
patch again. Its CMake pins Nicaea to
`f7850e5009d0341226d4c1f38b116f316d8a0516` and Sparse2D to `v2.1.2`.
The recorded build requires CMake, C/C++ compilers, Boost program_options, GSL,
FFTW, NFFT, CFITSIO and CCFits. Preserve the upstream CeCILL license with this tree.
The old Nicaea patch and legacy `pyfits` converter are not used by this build/driver
workflow and were omitted.

## Configure paths and launch a driver

Edit `config_paths.toml`, or create a separate TOML with these four keys:

```toml
data_root = "data"
work_root = "work"
glimpse_root = "external/glimpse"
mcalens_root = "external/cosmostat"
```

Relative paths resolve from the TOML file's directory, independently of the
shell's current directory. Absolute paths are also supported. For Windows paths,
use forward slashes or TOML single-quoted literal strings. The four locations are
raw FAIR files, generated experiment products, the GLIMPSE source/build checkout,
and the patched CosmoStat checkout. They are not scientific parameter settings.
The default target remains the original Linux scientific environment; GLIMPSE's
native build has not been validated on Windows.

The exact archived prior is read directly from `config/priors/`. Its generator
writes regenerated estimates under `work_root/validation/mcalens/`; this does not
silently replace the frozen input. Archived CSVs under `results/` likewise remain
separate from regenerated outputs. With an existing historical server layout,
`work_root` and `glimpse_root` may point to the same GLIMPSE directory.

Use the launcher for both pipeline and GLIMPSE drivers so that local scientific
modules and the configured CosmoStat checkout are on the Python import path:

```bash
python tools/run_pipeline.py --list
python tools/run_pipeline.py --dry-run run_square_starlet_comparison.py --split validation
python tools/run_pipeline.py --paths /path/to/config_paths.toml run_square_starlet_comparison.py --split validation
```

The last command runs the existing full validation driver. It is not a one-case
smoke test. Place launcher options before the script name; arguments after the
name are passed unchanged to that script. `WL_PATHS_CONFIG` is the equivalent
environment setting. Child processes inherit the same configuration and module
locations. Reading configuration or using `--dry-run` creates no data products.

The complete pre-edit list, source diff and automatic checks are under
`provenance/portability/`. To inspect preservation and prerequisite checks:

```bash
python tools/verify_scientific_invariants.py
python tools/check_portability.py
python tools/smoke_preflight.py
```

The source guard verifies all 25 original scripts, all 3,091 numeric literals,
and the remaining 94 original files. It restores only the explicitly planned
path expressions in an AST copy and compares the entire result with the v2
baseline hash. The patch is `scientific_path_changes.patch`; numerical solver
equivalence across environments has not been inferred from these static checks.

## Reproduction order

This is the intended dependency order, not a claim of a completed end-to-end
rerun. All driver paths now use central configuration. The historical paths in
provenance, recorded environments and CSV metadata are preserved as evidence.
Many drivers perform work at import time. Use the server wrapper with
`--preflight-only` for actual per-environment prerequisite imports. The lower-level
`smoke_preflight.py` performs static checks only.

1. Obtain the three raw FAIR files in `data_root` and check hashes. The included
   prior is resolved from this repository. The split is C0–C2, R0–R9 validation
   and R10–R29 test, with eight 176×176 patches per map.
2. Prepare validation observations with
   `code/glimpse/prepare_glimpse_validation.py`. Run
   `code/pipeline/run_square_starlet_comparison.py --split validation` and
   `--split test` for KS, mask-aware Starlet and the no-mask ablation. These produce
   the cached observations consumed by subsequent stages. To regenerate the prior,
   run `code/pipeline/build_fair_validation_pk.py` after the validation cache exists,
   writing to a separate work location and comparing against the frozen array.
3. Run `code/glimpse/run_glimpse_validation.py` and
   `code/glimpse/run_glimpse_test_final.py`, followed by
   `audit_glimpse_fits_final.py` and `bootstrap_glimpse_comparison.py`.
4. Validate Gaussian+Starlet with `run_gaussian_starlet_validation.py` and
   `run_gaussian_starlet_joint_validation.py`. Generate its validation map/component
   cache through stage 1 of `run_reweighted_gaussian_starlet_validation.py`.
5. Run `generate_mcalens_validation_maps.py` for the frozen Nsigma=5 validation
   maps and `run_mcalens_test.py` for the 480 test maps (R10–R29). The recovered
   `run_mcalens_validation.py` is a separate resumable threshold runner currently
   saved with `NSIGMAS=[6]`; it is not a complete fresh-run recipe for the original
   threshold selection. Its resume CSV expects `Nsigma`, whereas the retained
   final validation metrics use `nsigma`. Keep those outputs separate. The saved
   validation metrics contain 240 Nsigma=5 cases only.
6. Run `run_gaussian_starlet_test.py` and `run_wiener_gaussian_only.py` using the
   frozen settings and observations. GS's final comparisons also read the supplied
   MCALens test metrics.
7. Run hard-band validation (`run_multiscale_fusion_validation.py`) and the weighted
   grid (`run_weighted_band_fusion_validation.py`), then the frozen test fusion
   (`run_weighted_band_fusion_test.py`). The paper uses `(0, .75, .75, 0)` with the
   GS coarse component. Preserve the documented choice rather than selecting anew.
8. Run `run_admm_gaussian_starlet_audit.py` for Appendix A,
   `bootstrap_wiener_vs_gs.py`, and
   `bootstrap_peak_prauc_v5_with_gs_mcalens_bootstrap.py` for quoted intervals.
9. Run `run_gaussian_starlet_reduced_shear.py` and `quantify_photoz_trend.py` for the
   robustness analyses. The four final GS reduced-shear CSVs are included under
   `results/test/gaussian_starlet_reduced_shear/`.
10. Run `make_revised_figures_v12.py` after the required raw inputs and reconstructed
    Wiener, GS, MCALens and fusion maps exist. It computes the final spectra/peak
    diagnostics as well as plotting. The four V12 CSVs allow table checks, but
    the unchanged generator still requires the map arrays to regenerate figures.

Unless otherwise prefixed, scripts in steps 4–10 are in `code/pipeline/`.
See `provenance/manuscript_map.csv` for the precise table/figure-to-file mapping.
Some scripts emit earlier ancillary summaries as well as required maps; canonical
final manuscript peak/spectral values are the V12 outputs under `results/paper/`.

## Validation and publication status

Archive hashes and all supplied follow-up manifest entries were checked. Every
non-transformed original matches its original SHA256; the 24 path-adjusted scripts
pass the full source guard. All 25 scientific Python files parse and compile.
The table checker passed all 162 checks, including complete
480-case grids for all eight methods, the final reduced-shear experiment, and the
30-case solver audit. No reconstruction,
C++ build, bootstrap rerun or figure rendering was performed in the local Windows
review environment.

`CITATION.cff` uses the actual manuscript title and authors. Rafael approved MIT
for his own contributions; upstream notices and license boundaries are preserved.
The code repository is hosted at
[rafaujo/weak-lensing-mass-mapping](https://github.com/rafaujo/weak-lensing-mass-mapping).
The manuscript source and figure PDFs are not included. A versioned release and
Zenodo archive remain to be prepared; the actual DOI will be added after archival
publication. Historical provenance files retain the status of their dated snapshots.
The initial repository metadata changes are recorded in
`provenance/publication/initial_publication.json`.
