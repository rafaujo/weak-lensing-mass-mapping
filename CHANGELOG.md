# Changelog

## v1.1.0 - 2026-10-02

- Include the validated convergence-zero peak analysis on all 480 saved test maps.
- Archive both conventions, counts, matched amplitudes, curves, offsets, bootstrap
  draws, run settings and the independent audit under `analysis/peak_zero/`.
- Add a NumPy-only checker for the archived summaries and paired intervals.
- Update citation metadata and reproduction instructions for this release.
- Preserve all previously published scientific scripts, configurations, inputs,
  numerical results and third-party licence notices unchanged.

The primary stored-zero results remain unchanged. This release supplies the
additional sensitivity analysis discussed in Appendix B; it does not introduce
a new method, retune hyperparameters, or rerun the reconstruction benchmark.

## v1.0.0 - 2026-10-02

Initial benchmark release, archived at https://doi.org/10.5281/zenodo.23102124.
