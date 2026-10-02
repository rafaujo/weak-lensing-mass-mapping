"""Helpers extracted without algorithm changes from the archived v1.0.0 code.
See helper_provenance.json. No reconstruction routines or top-level I/O.
"""

import numpy as np

from scipy.ndimage import gaussian_filter, maximum_filter

from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import maximum_bipartite_matching

MASK_SUPPORT_MIN = 0.70

def mask_normalized_smooth(image, mask, sigma_pix):
    image = np.asarray(image, dtype=np.float64)
    mask_float = mask.astype(np.float64)
    numerator = gaussian_filter(image * mask_float, sigma=sigma_pix, mode='constant', cval=0.0)
    denominator = gaussian_filter(mask_float, sigma=sigma_pix, mode='constant', cval=0.0)
    valid = denominator >= MASK_SUPPORT_MIN
    output = np.zeros_like(numerator)
    output[valid] = numerator[valid] / denominator[valid]
    return (output, valid)

def find_peaks(field, valid):
    local_max = maximum_filter(field, size=3, mode='constant', cval=-np.inf)
    is_peak = valid & np.isfinite(field) & (field == local_max)
    coordinates = np.argwhere(is_peak)
    values = field[is_peak]
    return (coordinates, values)

def exact_match_count(truth_coords, rec_coords, radius_pix):
    n_truth = len(truth_coords)
    n_rec = len(rec_coords)
    if n_truth == 0 or n_rec == 0:
        return 0
    dy = truth_coords[:, None, 0] - rec_coords[None, :, 0]
    dx = truth_coords[:, None, 1] - rec_coords[None, :, 1]
    adjacency = dx ** 2 + dy ** 2 <= radius_pix ** 2
    if not np.any(adjacency):
        return 0
    graph = csr_matrix(adjacency.astype(np.int8))
    matching = maximum_bipartite_matching(graph, perm_type='column')
    return int(np.sum(matching >= 0))

def nearest_amplitude_pairs(truth_coords, rec_coords, radius_pix):
    if len(truth_coords) == 0 or len(rec_coords) == 0:
        return []
    dy = truth_coords[:, None, 0] - rec_coords[None, :, 0]
    dx = truth_coords[:, None, 1] - rec_coords[None, :, 1]
    d2 = dx ** 2 + dy ** 2
    candidate_pairs = np.argwhere(d2 <= radius_pix ** 2)
    if len(candidate_pairs) == 0:
        return []
    distance_values = d2[candidate_pairs[:, 0], candidate_pairs[:, 1]]
    order = np.argsort(distance_values)
    used_truth = set()
    used_rec = set()
    pairs = []
    for kk in order:
        ti = int(candidate_pairs[kk, 0])
        ri = int(candidate_pairs[kk, 1])
        if ti in used_truth or ri in used_rec:
            continue
        used_truth.add(ti)
        used_rec.add(ri)
        pairs.append((ti, ri))
    return pairs

def canonicalize(arr, name):
    arr = np.asarray(arr)
    if arr.shape == (3, 20, 8, 176, 176):
        return arr.astype(np.float32, copy=False)
    if arr.shape == (3, 30, 8, 176, 176):
        return arr[:, 10:30, :, :, :].astype(np.float32, copy=False)
    if arr.shape == (480, 176, 176):
        return arr.reshape(3, 20, 8, 176, 176).astype(np.float32, copy=False)
    raise ValueError(f'{name}: unsupported shape {arr.shape}')

def curve_from_counts(truth_total, predicted, matched):
    completeness = matched / max(truth_total, 1)
    purity = np.divide(matched, predicted, out=np.full(len(predicted), np.nan, dtype=float), where=predicted > 0)
    return (completeness, purity)

def normalized_common_auc(curves):
    valid_ranges = []
    for method, (completeness, purity) in curves.items():
        valid = np.isfinite(completeness) & np.isfinite(purity)
        if np.sum(valid) < 2:
            return {m: np.nan for m in curves}
        valid_ranges.append((np.min(completeness[valid]), np.max(completeness[valid])))
    common_lo = max((x[0] for x in valid_ranges))
    common_hi = min((x[1] for x in valid_ranges))
    if common_hi <= common_lo:
        return {method: np.nan for method in curves}
    output = {}
    for method, (completeness, purity) in curves.items():
        valid = np.isfinite(completeness) & np.isfinite(purity) & (completeness >= common_lo) & (completeness <= common_hi)
        cc = completeness[valid]
        pp = purity[valid]
        order = np.argsort(cc)
        cc = cc[order]
        pp = pp[order]
        unique_c = np.unique(cc)
        unique_p = []
        for value in unique_c:
            indices = np.where(cc == value)[0]
            unique_p.append(pp[indices[-1]])
        unique_p = np.asarray(unique_p)
        if len(unique_c) < 2:
            output[method] = np.nan
        else:
            output[method] = np.trapz(unique_p, unique_c) / (common_hi - common_lo)
    return output
