#!/usr/bin/env python3
"""Reanalyse saved maps, first reproducing the published peak diagnostics.

Only the convergence-zero convention changes in the second analysis. No solvers,
retuning, new noise or changes to the historical peak/matching/AUC routines.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import time

import numpy as np
import scipy
from historical_helpers import (
    canonicalize, mask_normalized_smooth, find_peaks, exact_match_count,
    nearest_amplitude_pairs, curve_from_counts, normalized_common_auc,
)

ROOT = Path(__file__).resolve().parent
METHODS = ('GAUSSIAN_STARLET', 'MCALENS', 'FUSION')
MODES = ('stored', 'observed_mean_zero')
SCALES = (2., 4., 6.)
THRESHOLDS = np.linspace(8., -.5, 120)
PAIRS = ((2, 0), (2, 1), (0, 1))
PIX_ARCMIN = 2.
TRUTH_NU_PR = 2.5
TRUTH_NU_AMP = 3.
MATCH_RADIUS_PIX = 2.
N_BOOT = 10000
SEED = 123456


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def prepare_image(image, mask, mode):
    image = np.asarray(image, dtype=np.float64)
    if mode == 'observed_mean_zero':
        image = image - image[mask].mean(dtype=np.float64)
    elif mode != 'stored':
        raise ValueError(mode)
    return image


def analyse_patch(truth, maps, mask, scale, mode):
    """Historical peak loop; the only new operation is prepare_image()."""
    ts, tv = mask_normalized_smooth(prepare_image(truth, mask, mode), mask, scale / PIX_ARCMIN)
    sigma_truth = np.std(ts[tv]) + 1e-30
    coords, vals = find_peaks(ts / sigma_truth, tv)
    tc = coords[vals >= TRUTH_NU_PR]
    ta = coords[vals >= TRUTH_NU_AMP]
    pred = np.zeros((3, len(THRESHOLDS)), dtype=np.int64)
    matched = np.zeros_like(pred)
    amp = []
    for mi, method in enumerate(METHODS):
        rs, rv = mask_normalized_smooth(prepare_image(maps[method], mask, mode), mask, scale / PIX_ARCMIN)
        rc, rh = find_peaks(rs / sigma_truth, tv & rv)
        # Identical recon catalogues at adjacent thresholds need only one match.
        cache = {}
        for ti, threshold in enumerate(THRESHOLDS):
            keep = rh >= threshold
            key = keep.tobytes()
            if key not in cache:
                cache[key] = (int(keep.sum()), exact_match_count(tc, rc[keep], MATCH_RADIUS_PIX))
            pred[mi, ti], matched[mi, ti] = cache[key]
        for a, b in nearest_amplitude_pairs(ta, rc, MATCH_RADIUS_PIX):
            yt, xt = ta[a]
            yr, xr = rc[b]
            truth_amp, rec_amp = ts[yt, xt], rs[yr, xr]
            if abs(truth_amp) > 1e-30:
                amp.append({'method': method, 'truth_y': int(yt), 'truth_x': int(xt),
                            'rec_y': int(yr), 'rec_x': int(xr), 'truth_amplitude': float(truth_amp),
                            'rec_amplitude': float(rec_amp), 'ratio': float(rec_amp / truth_amp)})
    return len(tc), pred, matched, amp


class SavedMaps:
    def __init__(self, args):
        self.mask_file = args.data_root / 'WIDE12H_bin2_2arcmin_mask.npy'
        self.truth_file = args.data_root / 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
        self.gs_file = args.gs_file or args.maps_root / 'test/starlet_gaussian_prior/gaussian_starlet_test_reconstruction.npy'
        self.fusion_file = args.fusion_file or args.maps_root / 'test/weighted_band_fusion/W_A1_075_A2_075_test_maps.npy'
        self.mc_root = args.mcalens_dir or args.maps_root / 'test/mcalens/recon'
        self.mc_key = args.mcalens_key
        if any(word in self.mc_key.lower() for word in ('truth', 'true', 'mask', 'gamma', 'shear', 'noise', 'input', 'target', 'label')):
            raise ValueError(f'Refusing non-reconstruction MCALens key: {self.mc_key!r}')
        self.files = [self.mask_file, self.truth_file, self.gs_file, self.fusion_file]
        self.files += [self.mc_root / f'C{c}_R{r}_P{p}.npz' for c in range(3) for r in range(10, 30) for p in range(8)]
        missing = [str(p) for p in self.files if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} required input files missing; first 12:\n' + '\n'.join(missing[:12]))
        self.mask_full = np.load(self.mask_file, allow_pickle=False).astype(bool)
        self.kappa = np.load(self.truth_file, mmap_mode='r', allow_pickle=False)
        if self.mask_full.shape != (1424, 176) or int(self.mask_full.sum()) != 132019:
            raise ValueError(f'Unexpected FAIR mask: {self.mask_full.shape}, {self.mask_full.sum()} observed pixels')
        if self.kappa.shape != (3, 30, 132019):
            raise ValueError(f'Unexpected truth shape: {self.kappa.shape}')
        self.gs = canonicalize(np.load(self.gs_file, mmap_mode='r', allow_pickle=False), 'GS')
        self.fusion = canonicalize(np.load(self.fusion_file, mmap_mode='r', allow_pickle=False), 'FUSION')
        # Use an explicit reconstruction key, never guess from shape or take truth.
        for path in self.files[4:]:
            with np.load(path, allow_pickle=False) as z:
                if self.mc_key not in z or z[self.mc_key].shape != (176, 176):
                    raise ValueError(f'{path}: expected reconstruction key {self.mc_key!r}, available {z.files}')
        self.metadata = {'n_patches': 480, 'mask_shape': list(self.mask_full.shape),
                         'truth_shape': list(self.kappa.shape), 'mcalens_key': self.mc_key,
                         'files': [{'path': str(p.resolve()), 'bytes': p.stat().st_size,
                                    'sha256': sha256(p)} for p in self.files]}

    def patches(self):
        for c in range(3):
            for r in range(10, 30):
                full = np.zeros(self.mask_full.shape, dtype=np.float32)
                full[self.mask_full] = np.asarray(self.kappa[c, r], dtype=np.float32)
                for p in range(8):
                    with np.load(self.mc_root / f'C{c}_R{r}_P{p}.npz', allow_pickle=False) as z:
                        mc = np.asarray(z[self.mc_key], dtype=np.float32)
                    sl = slice(p * 176, (p + 1) * 176)
                    maps = dict(zip(METHODS, (self.gs[c, r-10, p], mc, self.fusion[c, r-10, p])))
                    truth, mask = full[sl], self.mask_full[sl]
                    for name, arr in [('TRUTH', truth), *maps.items()]:
                        if not np.isfinite(arr).all():
                            raise ValueError(f'Nonfinite {name} C{c}/R{r}/P{p}')
                    yield c, r, p, truth, maps, mask


def curves_from_blocks(truth, pred, matched, indices):
    return {m: curve_from_counts(int(truth[indices].sum()), pred[i, indices].sum(axis=0),
                                matched[i, indices].sum(axis=0)) for i, m in enumerate(METHODS)}


def common_interval(curves):
    intervals = []
    for comp, pur in curves.values():
        valid = np.isfinite(comp) & np.isfinite(pur)
        if valid.sum() < 2:
            raise ValueError('Insufficient points for common completeness interval')
        intervals.append((float(comp[valid].min()), float(comp[valid].max())))
    lo, hi = max(x[0] for x in intervals), min(x[1] for x in intervals)
    if hi <= lo:
        raise ValueError('No common completeness interval')
    return lo, hi


def run_mode(loader, mode, output):
    truth_counts = np.zeros((3, 20), dtype=np.int64)
    pred = np.zeros((3, 3, 20, 120), dtype=np.int64)
    matched = np.zeros_like(pred)
    amps, offsets = [], []
    for case, (c, r, p, truth, maps, mask) in enumerate(loader.patches(), 1):
        for name, arr in [('TRUTH', truth), *maps.items()]:
            offsets.append({'c': c, 'r': r, 'p': p, 'method': name,
                            'observed_mean': float(np.asarray(arr, dtype=np.float64)[mask].mean()),
                            'mean_after': float(prepare_image(arr, mask, mode)[mask].mean())})
        for si, scale in enumerate(SCALES):
            nt, pr, ma, ar = analyse_patch(truth, maps, mask, scale, mode)
            truth_counts[si, r-10] += nt
            pred[si, :, r-10] += pr
            matched[si, :, r-10] += ma
            amps.extend({'c': c, 'r': r, 'p': p, 'scale_arcmin': scale, **row} for row in ar)
        if case % 24 == 0:
            print(f'{mode}: {case}/480 patches', flush=True)
    np.savez_compressed(output / f'{mode}_block_counts.npz', truth=truth_counts,
                        predicted=pred, matched=matched, scales=SCALES,
                        thresholds=THRESHOLDS, methods=METHODS, realisations=np.arange(10, 30))
    write_csv(output / f'{mode}_amplitude_pairs.csv', amps)
    write_csv(output / f'{mode}_offsets.csv', offsets)
    summaries, curve_rows = [], []
    for si, scale in enumerate(SCALES):
        curves = curves_from_blocks(truth_counts[si], pred[si], matched[si], np.arange(20))
        auc = normalized_common_auc(curves)
        lo, hi = common_interval(curves)
        for method in METHODS:
            ratios = [a['ratio'] for a in amps if a['scale_arcmin'] == scale and a['method'] == method]
            if not ratios or not np.isfinite(auc[method]):
                raise ValueError(f'Insufficient peaks: {mode}/{scale}/{method}')
            summaries.append({'mode': mode, 'scale_arcmin': scale, 'method': method,
                              'auc_common': float(auc[method]),
                              'peak_amplitude_ratio': float(np.median(ratios)),
                              'n_amplitude_pairs': len(ratios), 'n_truth_peaks': int(truth_counts[si].sum()),
                              'common_completeness_min': lo, 'common_completeness_max': hi})
            comp, pur = curves[method]
            for i, threshold in enumerate(THRESHOLDS):
                curve_rows.append({'scale_arcmin': scale, 'method': method, 'threshold_rec': threshold,
                                   'completeness': comp[i], 'purity': pur[i], 'auc_common': auc[method],
                                   'common_completeness_min': lo, 'common_completeness_max': hi})
    write_csv(output / f'{mode}_summary.csv', summaries)
    write_csv(output / f'{mode}_curves.csv', curve_rows)
    return truth_counts, pred, matched, summaries, curve_rows


def compare_baseline(summaries, curves):
    checks = []
    expected = read_csv(ROOT / 'reference/peak_completeness_purity.csv')
    if len(curves) != len(expected):
        raise ValueError('Baseline curve row count differs')
    for actual, frozen in zip(curves, expected):
        if actual['method'] != frozen['method'] or actual['scale_arcmin'] != float(frozen['scale_arcmin']):
            raise ValueError('Baseline curve ordering differs')
        for key in ('threshold_rec', 'completeness', 'purity', 'auc_common', 'common_completeness_min', 'common_completeness_max'):
            got, want = float(actual[key]), float(frozen[key])
            ok = bool(np.isclose(got, want, rtol=0, atol=1e-10, equal_nan=True))
            checks.append({'scale': actual['scale_arcmin'], 'method': actual['method'],
                           'metric': key, 'threshold': float(actual['threshold_rec']),
                           'absolute_error': abs(got-want) if np.isfinite(got-want) else None, 'pass': ok})
    for frozen in read_csv(ROOT / 'reference/peak_amplitude.csv'):
        row = next(x for x in summaries if x['method'] == frozen['method'] and x['scale_arcmin'] == float(frozen['scale_arcmin']))
        err = abs(row['peak_amplitude_ratio'] - float(frozen['peak_amplitude_ratio']))
        checks.append({'scale': row['scale_arcmin'], 'method': row['method'],
                       'metric': 'amplitude_ratio', 'absolute_error': err, 'pass': err <= 1e-8})
    return {'passed': all(x['pass'] for x in checks), 'n_checks': len(checks),
            'n_failures': sum(not x['pass'] for x in checks),
            'max_finite_error': max(x['absolute_error'] or 0 for x in checks),
            'failures': [x for x in checks if not x['pass']][:100]}


def bootstrap(results, output):
    rng = np.random.default_rng(SEED)
    rows = []
    for si, scale in enumerate(SCALES):
        arrays = {mode: np.empty((N_BOOT, 3), dtype=float) for mode in MODES}
        points = {}
        for mode, (nt, pr, ma, _, _) in results.items():
            points[mode] = normalized_common_auc(curves_from_blocks(nt[si], pr[si], ma[si], np.arange(20)))
        for b in range(N_BOOT):
            indices = rng.integers(0, 20, size=20)
            # Exactly the same realisation-block draw for both zero conventions.
            for mode, (nt, pr, ma, _, _) in results.items():
                vals = normalized_common_auc(curves_from_blocks(nt[si], pr[si], ma[si], indices))
                arrays[mode][b] = [vals[m] for m in METHODS]
            if (b + 1) % 2000 == 0:
                print(f'bootstrap {scale:g} arcmin: {b+1}/{N_BOOT}', flush=True)
        for mode in MODES:
            for a, b in PAIRS:
                delta = arrays[mode][:, a] - arrays[mode][:, b]
                good = np.isfinite(delta)
                if not good.all():
                    raise ValueError('Nonfinite bootstrap replicate; inspect the saved counts')
                lo, hi = np.percentile(delta, [2.5, 97.5])
                m1, m2 = METHODS[a], METHODS[b]
                rows.append({'mode': mode, 'scale_arcmin': scale, 'method_1': m1, 'method_2': m2,
                             'auc_1': points[mode][m1], 'auc_2': points[mode][m2],
                             'delta': points[mode][m1]-points[mode][m2],
                             'ci95_low': lo, 'ci95_high': hi,
                             'bootstrap_fraction_gt0': float(np.mean(delta > 0)),
                             'n_blocks': 20, 'n_boot': N_BOOT})
        np.savez_compressed(output / f'bootstrap_{scale:g}arcmin.npz', **arrays, methods=METHODS)
    write_csv(output / 'bootstrap_comparisons.csv', rows)
    checks = []
    for ref in read_csv(ROOT / 'reference/peak_prauc_block_bootstrap.csv'):
        row = next(r for r in rows if r['mode']=='stored' and r['scale_arcmin']==float(ref['scale_arcmin'])
                   and r['method_1']==ref['method_1'] and r['method_2']==ref['method_2'])
        for key in ('delta', 'ci95_low', 'ci95_high', 'bootstrap_fraction_gt0'):
            err = abs(float(row[key])-float(ref[key]))
            checks.append({'scale': row['scale_arcmin'], 'pair': [row['method_1'], row['method_2']],
                           'metric': key, 'absolute_error': err, 'pass': err < 1e-8})
    check = {'passed': all(x['pass'] for x in checks), 'checks': checks}
    write_json(output / 'baseline_bootstrap_check.json', check)
    if not check['passed']:
        raise ValueError('Stored-map bootstrap did not reproduce the archived intervals')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path('/tatu/gob5/99 - Cosmos/data'))
    ap.add_argument('--maps-root', type=Path, default=Path('/u/gob5/Glimpse'))
    ap.add_argument('--gs-file', type=Path)
    ap.add_argument('--fusion-file', type=Path)
    ap.add_argument('--mcalens-dir', type=Path)
    ap.add_argument('--mcalens-key', default='mcalens')
    ap.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused')
    ap.add_argument('--preflight', action='store_true', help='Validate and hash all inputs, without peak analysis')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        loader = SavedMaps(args)
        write_json(args.output / 'input_manifest.json', loader.metadata)
        write_json(args.output / 'run_configuration.json', {
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'command': sys.argv, 'scales_arcmin': SCALES, 'pixel_arcmin': PIX_ARCMIN,
            'truth_nu_pr': TRUTH_NU_PR, 'truth_nu_amp': TRUTH_NU_AMP, 'radius_pixels': MATCH_RADIUS_PIX,
            'mask_support_min': .70, 'reconstruction_thresholds': THRESHOLDS.tolist(),
            'seed': SEED, 'n_boot': N_BOOT, 'mode_definitions': {
                'stored': 'Original maps, no mean subtraction',
                'observed_mean_zero': 'Subtract each map own observed-pixel mean, per patch, before smoothing; truth included'},
            'code_sha256': {p.name: sha256(p) for p in [Path(__file__), ROOT / 'historical_helpers.py']}})
        if args.preflight:
            write_json(args.output / 'status.json', {'status': 'PREFLIGHT_OK', 'n_patches': 480})
            print('PREFLIGHT_OK: all 480 cases available. No reconstructions run.', flush=True)
            return
        results = {'stored': run_mode(loader, 'stored', args.output)}
        baseline = compare_baseline(results['stored'][3], results['stored'][4])
        write_json(args.output / 'baseline_check.json', baseline)
        if not baseline['passed']:
            raise ValueError('BASELINE_MISMATCH: stop; do not interpret centered results')
        print('BASELINE_OK: published curves, AUCs and amplitudes reproduced.', flush=True)
        results['observed_mean_zero'] = run_mode(loader, 'observed_mean_zero', args.output)
        boot = bootstrap(results, args.output)
        comparisons = []
        for si, scale in enumerate(SCALES):
            ranks = {}
            for mode in MODES:
                subset = [r for r in results[mode][3] if r['scale_arcmin']==scale]
                ranks[mode] = [r['method'] for r in sorted(subset, key=lambda r:r['auc_common'], reverse=True)]
            comparisons.append({'scale_arcmin': scale, 'auc_ranking': ranks,
                                'ranking_unchanged': ranks['stored']==ranks['observed_mean_zero']})
        status = {'status': 'COMPLETE', 'n_patches': 480, 'baseline_passed': True,
                  'baseline_bootstrap_passed': True, 'elapsed_seconds': time.time()-started,
                  'rankings': comparisons,
                  'interpretation': 'Compare centered paired CIs, not point rankings alone. Common completeness ranges are recalculated separately in each mode. Amplitude medians condition on matched peaks; the selected truth catalogues may change. No cosmological inference performed.'}
        write_json(args.output / 'status.json', status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_json(args.output / 'status.json', {'status': 'FAILED', 'error': str(exc), 'elapsed_seconds': time.time()-started})
        raise


if __name__ == '__main__':
    main()
