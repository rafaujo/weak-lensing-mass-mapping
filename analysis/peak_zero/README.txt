CONVERGENCE-ZERO SENSITIVITY OF PEAK STATISTICS - RELEASE v1.1.0
Companion to: Sparse and Gaussian Priors for Weak-Lensing Mass Mapping:
Observable-Dependent Performance on FAIR Universe Simulations
Rafael L. de Araujo and Reynam da C. Pestana
This analysis supports Appendix B and is included in release v1.1.0 of
https://github.com/rafaujo/weak-lensing-mass-mapping, under analysis/peak_zero/.
It was not part of the earlier v1.0.0 archive (doi:10.5281/zenodo.23102124).
Code, frozen results and the original audit evidence are preserved from the
validated package. See the repository README for the current archival citation.
FAIR source data: https://doi.org/10.5281/zenodo.20056065.
No reconstruction was rerun. We used 480 saved test patches, covering the
same three cosmologies, R10--R29 and eight patches per realisation. The code
first reproduces the archived stored-map diagnostics, then subtracts each
map's own observed-pixel mean before mask-normalised smoothing. It repeats
peak detection, matching, amplitude medians and paired realisation bootstrap.
REPRODUCING THE ANALYSIS
Run the commands below from analysis/peak_zero/.
Use Python 3.12, NumPy 1.26.4 and SciPy 1.16.2 (requirements-tested.txt).
python code/peak_zero_test.py --data-root /path/to/FAIR/data \
    --maps-root /path/to/work --output /path/to/new_output_directory
The maps-root must contain:
test/starlet_gaussian_prior/gaussian_starlet_test_reconstruction.npy
test/weighted_band_fusion/W_A1_075_A2_075_test_maps.npy
test/mcalens/recon/C{c}_R{r}_P{p}.npz (reconstruction key: mcalens)
Use --gs-file, --fusion-file or --mcalens-dir to override those locations.
The raw FAIR inputs are WIDE12H_bin2_2arcmin_mask.npy and
sampled_WIDE12H_bin2_2arcmin_kappa.npy. The large raw and reconstructed map
cubes are not duplicated in this analysis package; the core benchmark supplies
their reconstruction generators. GLIMPSE and CosmoStat are not imported
when running this post-processing analysis. Existing output directories
are refused, and inputs are read without modification.
RESULTS
The full server run completed in 168.869 seconds. The 6489 stored-map checks
and 36 original-bootstrap checks match the archived files exactly. The
independent audit recomputes 18 AUC/amplitude summaries from the block counts
and matched samples, and 18 confidence intervals from saved bootstrap draws.
Both modes use the same 10000 draws of 20 realisation blocks, seed 123456.
Detection point rankings are unchanged. All fusion contrasts and the GS-MC
contrasts at 2 and 4 arcmin keep their signs and individual 95% exclusions
of zero. At 6 arcmin, the centered GS-minus-MC delta is -0.0140324114 with
95% interval [-0.0295526755,+0.0015232531], which includes zero. The 2-arcmin
matched amplitude medians reverse the fusion/MC ordering: centered fusion
0.5563859032 versus MC 0.5447361054. MC remains closest to unity at 4 and
6 arcmin. Amplitude comparisons are descriptive; no amplitude CIs were made.
Mean removal changes the truth catalogue and the common completeness range.
The absolute AUC increase between modes is therefore not an improvement of
the saved reconstructions. Compare methods within each convention. This is
an additive-zero test for the linear-shear benchmark, not a test of the full
reduced-shear mass-sheet degeneracy or of cosmological parameter inference.
FILE GUIDE
results/*_summary.csv: point estimates in the stored and centered conventions.
results/bootstrap_comparisons.csv: all 18 paired contrasts and 95% intervals.
results/*_block_counts.npz: truth counts and 120-threshold matched/predicted
counts, indexed by scale, method and the 20 realisation blocks.
results/bootstrap_*arcmin.npz: paired AUC draws (10000 x 3) in both conventions.
results/*_amplitude_pairs.csv: selected/matched peaks, locations and ratios.
results/*_offsets.csv: original means and means after the selected operation.
results/*_curves.csv: completeness-purity curves and common ranges.
results/run_configuration.json: environment, settings and source hashes.
results/input_manifest.json: paths and hashes of the 484 input files.
results/baseline*check.json: comparisons with the frozen benchmark outputs.
code/helper_provenance.json: source locations and AST-identity checks for the
seven historical helper functions. Full historical sources in code/reference
are provenance records, not executed as reconstruction programs by this test.
validation/: server log and independent consistency audit.
The code licence is in code/LICENSE. The third-party FAIR data and external
software retain their own licences; no external reconstruction code is added
by this analysis package. MANIFEST.json lists hashes of all supplied files.
