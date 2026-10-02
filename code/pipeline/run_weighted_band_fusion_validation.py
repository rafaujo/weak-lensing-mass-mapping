from project_paths import PATHS
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
)
from scipy.spatial import cKDTree


# =============================================================================
# CONFIGURATION
# =============================================================================
N = 176
J = 4

PIXEL_ARCMIN = 2.0

BASE = Path(
    str(PATHS.work_root / 'validation/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

GS_FILE = Path(
    str(PATHS.work_root / 'validation/reweighted_gaussian_starlet/stage1_gaussian_starlet_rec.npy')
)

MC_DIR = Path(
    str(PATHS.work_root / 'validation/mcalens/recon')
)

OUT = Path(
    str(PATHS.work_root / 'validation/weighted_band_fusion')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PATCH_ROWS = [
    (0, 176),
    (176, 352),
    (352, 528),
    (528, 704),
    (704, 880),
    (880, 1056),
    (1056, 1232),
    (1232, 1408),
]

GRID_SHAPE = (
    3,
    10,
    8,
    N,
    N,
)

FLAT_SHAPE = (
    240,
    N,
    N,
)


# =============================================================================
# WEIGHT GRID
# =============================================================================
#
# j0  -> always Gaussian+Starlet
# j1  -> continuous GS/MC mixture
# j2  -> continuous GS/MC mixture
# j3  -> always Gaussian+Starlet
# coarse residual -> always Gaussian+Starlet
#
A1_GRID = [
    0.50,
    0.75,
    1.00,
]

A2_GRID = [
    0.50,
    0.75,
    1.00,
]


def candidate_name(
    a1,
    a2,
):
    return (
        f"W_A1_{int(round(100*a1)):03d}"
        f"_A2_{int(round(100*a2)):03d}"
    )


CANDIDATES = [
    (
        candidate_name(
            a1,
            a2,
        ),
        a1,
        a2,
    )
    for a1 in A1_GRID
    for a2 in A2_GRID
]

BASELINES = [
    "GAUSSIAN_STARLET",
    "MCALENS",
]

METHODS = (
    BASELINES
    + [
        name
        for name, _, _
        in CANDIDATES
    ]
)


# =============================================================================
# REFERENCE SANITY VALUES
# =============================================================================
EXPECTED_GS_NMSE = (
    0.6469240253547854
)

EXPECTED_GS_PCC = (
    0.5927316056309856
)

EXPECTED_MC_NMSE = (
    0.658007019
)

EXPECTED_MC_PCC = (
    0.591189437
)


# =============================================================================
# SPECTRAL CONFIG
# =============================================================================
N_ELL_BINS = 12

ELL_MIN = 90.0
ELL_MAX = 3000.0


# =============================================================================
# PEAK CONFIG
# =============================================================================
SMOOTH_ARCMIN = [
    2.0,
    4.0,
    6.0,
]

TRUTH_NU_THRESHOLDS = [
    2.0,
    2.5,
    3.0,
]

REC_NU_THRESHOLDS = np.arange(
    0.5,
    3.5001,
    0.25,
)

MATCH_RADIUS_ARCMIN = 4.0

MATCH_RADIUS_PIX = (
    MATCH_RADIUS_ARCMIN
    / PIXEL_ARCMIN
)

MIN_SMOOTH_SUPPORT = 0.95
EDGE_SIGMA = 3.0


# =============================================================================
# METRICS
# =============================================================================
def ms_metrics(
    rec,
    truth,
    mask,
):
    m = mask.astype(bool)

    rr = np.asarray(
        rec,
        dtype=np.float64,
    )[m]

    tt = np.asarray(
        truth,
        dtype=np.float64,
    )[m]

    # mass-sheet invariant
    rr -= rr.mean()
    tt -= tt.mean()

    mse = np.mean(
        (rr - tt) ** 2
    )

    nmse = (
        mse
        / np.mean(
            tt**2
        )
    )

    pcc = np.corrcoef(
        rr,
        tt,
    )[0, 1]

    std_ratio = (
        np.std(rr)
        / np.std(tt)
    )

    return (
        float(mse),
        float(nmse),
        float(pcc),
        float(std_ratio),
    )


# =============================================================================
# EXACT FOURIER STARLET
# =============================================================================
def build_starlet_filters(
    H,
    W,
    J,
):
    wy = (
        2.0
        * np.pi
        * np.fft.fftfreq(H)
    )

    wx = (
        2.0
        * np.pi
        * np.fft.fftfreq(W)
    )

    WY, WX = np.meshgrid(
        wy,
        wx,
        indexing="ij",
    )

    cumulative = np.ones(
        (H, W),
        dtype=np.float64,
    )

    bands = []

    for j in range(J):

        step = 2**j

        hy = (
            6.0
            + 8.0
            * np.cos(
                step * WY
            )
            + 2.0
            * np.cos(
                2.0
                * step
                * WY
            )
        ) / 16.0

        hx = (
            6.0
            + 8.0
            * np.cos(
                step * WX
            )
            + 2.0
            * np.cos(
                2.0
                * step
                * WX
            )
        ) / 16.0

        h2 = hy * hx

        bands.append(
            cumulative
            * (
                1.0
                - h2
            )
        )

        cumulative = (
            cumulative
            * h2
        )

    bands = np.asarray(
        bands,
        dtype=np.float64,
    )

    coarse = cumulative

    closure = (
        np.sum(
            bands,
            axis=0,
        )
        + coarse
    )

    closure_error = float(
        np.max(
            np.abs(
                closure
                - 1.0
            )
        )
    )

    return (
        bands,
        coarse,
        closure_error,
    )


BANDS_F, COARSE_F, CLOSURE_ERROR = (
    build_starlet_filters(
        N,
        N,
        J,
    )
)


def starlet_decompose_batch(
    maps,
):
    X = np.fft.fft2(
        maps,
        axes=(-2, -1),
        norm="ortho",
    )

    details = []

    for j in range(J):

        d = np.fft.ifft2(
            X
            * BANDS_F[
                j
            ][
                None,
                :,
                :,
            ],
            axes=(-2, -1),
            norm="ortho",
        ).real

        details.append(
            d.astype(
                np.float32
            )
        )

    details = np.stack(
        details,
        axis=1,
    )

    coarse = np.fft.ifft2(
        X
        * COARSE_F[
            None,
            :,
            :,
        ],
        axes=(-2, -1),
        norm="ortho",
    ).real.astype(
        np.float32
    )

    return (
        details,
        coarse,
    )


# =============================================================================
# LOAD VALIDATION TRUTH + MASK
# =============================================================================
print()
print("=" * 120)
print(
    "WEIGHTED STARLET-BAND FUSION — VALIDATION"
)
print("=" * 120)

print(
    "Starlet closure error:",
    f"{CLOSURE_ERROR:.3e}"
)

if CLOSURE_ERROR > 1e-12:
    raise RuntimeError(
        "Starlet decomposition does not close."
    )

mask_full = np.load(
    MASK_FILE
).astype(bool)

truths = []
masks = []
meta = []

for c in range(3):

    for r in range(10):

        fn = (
            BASE
            / f"C{c}_R{r}.npz"
        )

        with np.load(fn) as z:

            truth_full = np.asarray(
                z["truth"],
                dtype=np.float32,
            )

        for p, (
            r0,
            r1,
        ) in enumerate(
            PATCH_ROWS
        ):

            truths.append(
                truth_full[
                    r0:r1,
                    :
                ].copy()
            )

            masks.append(
                mask_full[
                    r0:r1,
                    :
                ].copy()
            )

            meta.append(
                (
                    c,
                    r,
                    p,
                )
            )

truths = np.asarray(
    truths,
    dtype=np.float32,
)

masks = np.asarray(
    masks,
    dtype=bool,
)

assert truths.shape == FLAT_SHAPE


# =============================================================================
# LOAD G+S
# =============================================================================
gs_grid = np.load(
    GS_FILE,
    mmap_mode="r",
)

if gs_grid.shape != GRID_SHAPE:
    raise RuntimeError(
        f"G+S shape mismatch: "
        f"{gs_grid.shape}"
    )

gs = np.empty(
    FLAT_SHAPE,
    dtype=np.float32,
)

for i, (
    c,
    r,
    p,
) in enumerate(meta):

    gs[i] = gs_grid[
        c,
        r,
        p,
    ]

del gs_grid


# =============================================================================
# LOAD MCALENS
# =============================================================================
mc = np.empty(
    FLAT_SHAPE,
    dtype=np.float32,
)

for i, (
    c,
    r,
    p,
) in enumerate(meta):

    fn = (
        MC_DIR
        / f"C{c}_R{r}_P{p}.npz"
    )

    if not fn.exists():
        raise FileNotFoundError(
            fn
        )

    with np.load(fn) as z:

        mc[i] = np.asarray(
            z["mcalens"],
            dtype=np.float32,
        )

print(
    "Loaded G+S    :",
    gs.shape,
)

print(
    "Loaded MCALens:",
    mc.shape,
)


# =============================================================================
# BASELINE SANITY
# =============================================================================
def summarize_maps(
    name,
    maps,
):
    vals = []

    for i in range(240):

        vals.append(
            ms_metrics(
                maps[i],
                truths[i],
                masks[i],
            )
        )

    vals = np.asarray(
        vals,
        dtype=np.float64,
    )

    return {
        "method": name,

        "nmse": float(
            vals[:, 1].mean()
        ),

        "pcc": float(
            vals[:, 2].mean()
        ),

        "std_ratio": float(
            vals[:, 3].mean()
        ),
    }


gs_check = summarize_maps(
    "GAUSSIAN_STARLET",
    gs,
)

mc_check = summarize_maps(
    "MCALENS",
    mc,
)

print()
print("=" * 120)
print("BASELINE SANITY")
print("=" * 120)

for x in [
    gs_check,
    mc_check,
]:

    print(
        f"{x['method']:20s} "
        f"NMSE={x['nmse']:.9f} "
        f"PCC={x['pcc']:.9f} "
        f"std={x['std_ratio']:.9f}"
    )

if abs(
    gs_check["nmse"]
    - EXPECTED_GS_NMSE
) > 2e-3:

    raise RuntimeError(
        "G+S validation mismatch."
    )

if abs(
    mc_check["nmse"]
    - EXPECTED_MC_NMSE
) > 2e-3:

    raise RuntimeError(
        "MCALens validation mismatch."
    )


# =============================================================================
# DECOMPOSE BOTH METHODS ONCE
# =============================================================================
print()
print("=" * 120)
print(
    "DECOMPOSING G+S AND MCALENS"
)
print("=" * 120)

gs_d = np.empty(
    (
        240,
        J,
        N,
        N,
    ),
    dtype=np.float32,
)

mc_d = np.empty_like(
    gs_d
)

gs_c = np.empty(
    FLAT_SHAPE,
    dtype=np.float32,
)

mc_c = np.empty_like(
    gs_c
)

BATCH = 16

max_gs_reconstruction_error = 0.0
max_mc_reconstruction_error = 0.0

for ib in range(
    0,
    240,
    BATCH,
):

    ie = min(
        ib + BATCH,
        240,
    )

    d1, c1 = (
        starlet_decompose_batch(
            gs[
                ib:ie
            ]
        )
    )

    d2, c2 = (
        starlet_decompose_batch(
            mc[
                ib:ie
            ]
        )
    )

    gs_d[
        ib:ie
    ] = d1

    gs_c[
        ib:ie
    ] = c1

    mc_d[
        ib:ie
    ] = d2

    mc_c[
        ib:ie
    ] = c2

    back_gs = (
        c1
        + np.sum(
            d1,
            axis=1,
        )
    )

    back_mc = (
        c2
        + np.sum(
            d2,
            axis=1,
        )
    )

    max_gs_reconstruction_error = max(
        max_gs_reconstruction_error,
        float(
            np.max(
                np.abs(
                    back_gs
                    - gs[
                        ib:ie
                    ]
                )
            )
        ),
    )

    max_mc_reconstruction_error = max(
        max_mc_reconstruction_error,
        float(
            np.max(
                np.abs(
                    back_mc
                    - mc[
                        ib:ie
                    ]
                )
            )
        ),
    )

    print(
        f"{ie:3d}/240",
        flush=True,
    )

print()
print(
    "max G+S recon error:",
    f"{max_gs_reconstruction_error:.3e}"
)

print(
    "max MC recon error :",
    f"{max_mc_reconstruction_error:.3e}"
)


# =============================================================================
# BUILD ALL WEIGHTED CANDIDATES
# =============================================================================
#
# Formula:
#
# h =
#   GS coarse
# + GS j0
# + [(1-a1) GS j1 + a1 MC j1]
# + [(1-a2) GS j2 + a2 MC j2]
# + GS j3
#
# Therefore a1=a2=1 reproduces H4.
#
# =============================================================================
maps = {
    "GAUSSIAN_STARLET": gs,
    "MCALENS": mc,
}

print()
print("=" * 120)
print(
    "BUILDING WEIGHTED CANDIDATES"
)
print("=" * 120)

for (
    name,
    a1,
    a2,
) in CANDIDATES:

    h = (
        gs_c
        + gs_d[
            :,
            0,
        ]
        + (
            (
                1.0
                - a1
            )
            * gs_d[
                :,
                1,
            ]
            +
            a1
            * mc_d[
                :,
                1,
            ]
        )
        + (
            (
                1.0
                - a2
            )
            * gs_d[
                :,
                2,
            ]
            +
            a2
            * mc_d[
                :,
                2,
            ]
        )
        + gs_d[
            :,
            3,
        ]
    ).astype(
        np.float32
    )

    maps[
        name
    ] = h

    np.save(
        OUT
        / f"{name}_validation_maps.npy",
        h,
    )

    print(
        f"{name:20s} "
        f"a1={a1:.2f} "
        f"a2={a2:.2f}"
    )


# =============================================================================
# FIELD METRICS
# =============================================================================
metric_rows = []

for method in METHODS:

    arr = maps[
        method
    ]

    for i, (
        c,
        r,
        p,
    ) in enumerate(meta):

        (
            mse,
            nmse,
            pcc,
            stdr,
        ) = ms_metrics(
            arr[i],
            truths[i],
            masks[i],
        )

        metric_rows.append(
            {
                "method": method,

                "c": c,
                "r": r,
                "p": p,

                "mse": mse,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": stdr,
            }
        )

metrics = pd.DataFrame(
    metric_rows
)

metrics.to_csv(
    OUT
    / "field_metrics_all.csv",
    index=False,
)

field_summary = (
    metrics
    .groupby(
        "method"
    )
    .agg(
        n=(
            "nmse",
            "size",
        ),

        nmse_mean=(
            "nmse",
            "mean",
        ),

        nmse_std=(
            "nmse",
            "std",
        ),

        nmse_median=(
            "nmse",
            "median",
        ),

        pcc_mean=(
            "pcc",
            "mean",
        ),

        std_ratio_mean=(
            "std_ratio",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "nmse_mean"
    )
)

field_summary.to_csv(
    OUT
    / "field_metrics_summary.csv",
    index=False,
)

print()
print("=" * 135)
print("FIELD-LEVEL VALIDATION SUMMARY")
print("=" * 135)

print(
    field_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =============================================================================
# BY COSMOLOGY
# =============================================================================
by_cosmo = (
    metrics
    .groupby(
        [
            "method",
            "c",
        ]
    )
    .agg(
        n=(
            "nmse",
            "size",
        ),

        nmse_mean=(
            "nmse",
            "mean",
        ),

        pcc_mean=(
            "pcc",
            "mean",
        ),

        std_ratio_mean=(
            "std_ratio",
            "mean",
        ),
    )
    .reset_index()
)

by_cosmo.to_csv(
    OUT
    / "field_by_cosmology.csv",
    index=False,
)


# =============================================================================
# PAIRED REALIZATION-BLOCK BOOTSTRAP
# =============================================================================
def paired_bootstrap(
    candidate,
    baseline,
    nboot=20000,
    seed=12345,
):
    a = metrics[
        metrics.method
        == candidate
    ][
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    b = metrics[
        metrics.method
        == baseline
    ][
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    a = a.rename(
        columns={
            "nmse": "nmse_a",
            "pcc": "pcc_a",
        }
    )

    b = b.rename(
        columns={
            "nmse": "nmse_b",
            "pcc": "pcc_b",
        }
    )

    q = a.merge(
        b,
        on=[
            "c",
            "r",
            "p",
        ],
        validate="one_to_one",
    )

    block = (
        q
        .groupby(
            "r"
        )
        .agg(
            nmse_a=(
                "nmse_a",
                "mean",
            ),

            nmse_b=(
                "nmse_b",
                "mean",
            ),

            pcc_a=(
                "pcc_a",
                "mean",
            ),

            pcc_b=(
                "pcc_b",
                "mean",
            ),
        )
        .sort_index()
    )

    na = block.nmse_a.to_numpy()
    nb = block.nmse_b.to_numpy()

    pa = block.pcc_a.to_numpy()
    pb = block.pcc_b.to_numpy()

    rng = np.random.default_rng(
        seed
    )

    inds = rng.integers(
        0,
        len(block),
        size=(
            nboot,
            len(block),
        ),
    )

    boot_nmse = (
        100.0
        * (
            nb[inds].mean(axis=1)
            - na[inds].mean(axis=1)
        )
        / nb[inds].mean(axis=1)
    )

    boot_pcc = (
        pa[inds].mean(axis=1)
        - pb[inds].mean(axis=1)
    )

    obs_nmse = (
        100.0
        * (
            nb.mean()
            - na.mean()
        )
        / nb.mean()
    )

    obs_pcc = (
        pa.mean()
        - pb.mean()
    )

    nlo, nhi = np.percentile(
        boot_nmse,
        [
            2.5,
            97.5,
        ],
    )

    plo, phi = np.percentile(
        boot_pcc,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "candidate": candidate,
        "baseline": baseline,

        "nmse_gain_pct": (
            obs_nmse
        ),

        "nmse_gain_ci_lo": (
            float(nlo)
        ),

        "nmse_gain_ci_hi": (
            float(nhi)
        ),

        "pcc_delta": (
            obs_pcc
        ),

        "pcc_ci_lo": (
            float(plo)
        ),

        "pcc_ci_hi": (
            float(phi)
        ),

        "block_wins_nmse": int(
            np.sum(
                na < nb
            )
        ),

        "block_total": int(
            len(block)
        ),
    }


boot_rows = []

for (
    name,
    _,
    _,
) in CANDIDATES:

    for baseline in [
        "GAUSSIAN_STARLET",
        "MCALENS",
    ]:

        boot_rows.append(
            paired_bootstrap(
                name,
                baseline,
            )
        )

field_boot = pd.DataFrame(
    boot_rows
)

field_boot.to_csv(
    OUT
    / "field_bootstrap.csv",
    index=False,
)


# =============================================================================
# SPECTRAL GRID
# =============================================================================
pix_rad = (
    PIXEL_ARCMIN
    / 60.0
    * np.pi
    / 180.0
)

ky = (
    2.0
    * np.pi
    * np.fft.fftfreq(
        N,
        d=pix_rad,
    )
)

kx = (
    2.0
    * np.pi
    * np.fft.fftfreq(
        N,
        d=pix_rad,
    )
)

KY, KX = np.meshgrid(
    ky,
    kx,
    indexing="ij",
)

ELL = np.sqrt(
    KX**2
    + KY**2
)

ELL_EDGES = np.geomspace(
    ELL_MIN,
    ELL_MAX,
    N_ELL_BINS + 1,
)


def centered_masked_map(
    x,
    mask,
):
    out = np.zeros_like(
        x,
        dtype=np.float64,
    )

    m = mask.astype(bool)

    out[m] = (
        x[m]
        - np.mean(
            x[m]
        )
    )

    return out


def spectral_case(
    truth,
    rec,
    mask,
):
    t = centered_masked_map(
        truth,
        mask,
    )

    r = centered_masked_map(
        rec,
        mask,
    )

    T = np.fft.fft2(
        t,
        norm="ortho",
    )

    R = np.fft.fft2(
        r,
        norm="ortho",
    )

    Ptt = np.abs(
        T
    ) ** 2

    Prr = np.abs(
        R
    ) ** 2

    Prt = np.real(
        R
        * np.conj(T)
    )

    rows = []

    for b in range(
        N_ELL_BINS
    ):

        sel = (
            (ELL >= ELL_EDGES[b])
            &
            (ELL < ELL_EDGES[b + 1])
        )

        if not np.any(
            sel
        ):
            continue

        tt = float(
            np.mean(
                Ptt[sel]
            )
        )

        rr = float(
            np.mean(
                Prr[sel]
            )
        )

        rt = float(
            np.mean(
                Prt[sel]
            )
        )

        if tt <= 0:
            continue

        T_cross = (
            rt / tt
        )

        A_auto = (
            rr / tt
        )

        r_ell = (
            rt
            / np.sqrt(
                max(
                    rr * tt,
                    1e-30,
                )
            )
        )

        rows.append(
            {
                "bin": b,

                "ell": float(
                    np.mean(
                        ELL[sel]
                    )
                ),

                "T_cross": (
                    T_cross
                ),

                "A_auto": (
                    A_auto
                ),

                "r_ell": (
                    r_ell
                ),
            }
        )

    return rows


# =============================================================================
# RUN SPECTRAL ANALYSIS
# =============================================================================
print()
print("=" * 120)
print("SPECTRAL ANALYSIS")
print("=" * 120)

spec_rows = []

for i, (
    c,
    r,
    p,
) in enumerate(meta):

    for method in METHODS:

        ss = spectral_case(
            truths[i],
            maps[method][i],
            masks[i],
        )

        for row in ss:

            row.update(
                {
                    "method": method,
                    "c": c,
                    "r": r,
                    "p": p,
                }
            )

            spec_rows.append(
                row
            )

    if (
        (i + 1)
        % 40
        == 0
    ):
        print(
            f"{i+1:3d}/240",
            flush=True,
        )

spec = pd.DataFrame(
    spec_rows
)

spec.to_csv(
    OUT
    / "spectral_all.csv",
    index=False,
)

spec_summary = (
    spec
    .groupby(
        [
            "method",
            "bin",
        ]
    )
    .agg(
        ell=(
            "ell",
            "mean",
        ),

        T_cross_mean=(
            "T_cross",
            "mean",
        ),

        A_auto_mean=(
            "A_auto",
            "mean",
        ),

        r_ell_mean=(
            "r_ell",
            "mean",
        ),
    )
    .reset_index()
)

spec_summary.to_csv(
    OUT
    / "spectral_summary.csv",
    index=False,
)


# =============================================================================
# COMPACT SPECTRAL SCORES
# =============================================================================
spectral_scores = []

for method in METHODS:

    q = spec_summary[
        spec_summary.method
        == method
    ]

    spectral_scores.append(
        {
            "method": method,

            "mean_abs_T_error": float(
                np.mean(
                    np.abs(
                        q.T_cross_mean
                        - 1.0
                    )
                )
            ),

            "mean_abs_A_error": float(
                np.mean(
                    np.abs(
                        q.A_auto_mean
                        - 1.0
                    )
                )
            ),

            "mean_r_ell": float(
                q.r_ell_mean.mean()
            ),
        }
    )

spectral_score = pd.DataFrame(
    spectral_scores
)

spectral_score.to_csv(
    OUT
    / "spectral_scores.csv",
    index=False,
)


# =============================================================================
# PEAK UTILITIES
# =============================================================================
def mask_normalized_smooth(
    image,
    mask,
    sigma_pix,
):
    m = mask.astype(
        np.float64
    )

    centered = np.zeros_like(
        image,
        dtype=np.float64,
    )

    obs = mask.astype(bool)

    centered[obs] = (
        image[obs]
        - np.mean(
            image[obs]
        )
    )

    numerator = gaussian_filter(
        centered * m,
        sigma=sigma_pix,
        mode="constant",
        cval=0.0,
    )

    denominator = gaussian_filter(
        m,
        sigma=sigma_pix,
        mode="constant",
        cval=0.0,
    )

    smoothed = np.zeros_like(
        numerator
    )

    good = (
        denominator
        > 1e-12
    )

    smoothed[good] = (
        numerator[good]
        / denominator[good]
    )

    valid = (
        denominator
        >= MIN_SMOOTH_SUPPORT
    )

    margin = int(
        np.ceil(
            EDGE_SIGMA
            * sigma_pix
        )
    )

    if margin > 0:

        valid[
            :margin,
            :
        ] = False

        valid[
            -margin:,
            :
        ] = False

        valid[
            :,
            :margin
        ] = False

        valid[
            :,
            -margin:
        ] = False

    return (
        smoothed,
        valid,
    )


def make_nu_map(
    image,
    mask,
    sigma_pix,
):
    sm, valid = (
        mask_normalized_smooth(
            image,
            mask,
            sigma_pix,
        )
    )

    if not np.any(
        valid
    ):
        raise RuntimeError(
            "No valid peak pixels."
        )

    sigma_map = float(
        np.std(
            sm[valid]
        )
    )

    if sigma_map <= 0:
        raise RuntimeError(
            "Invalid peak-map sigma."
        )

    return (
        sm,
        sm / sigma_map,
        valid,
    )


def peak_catalog(
    nu,
    valid,
    threshold,
):
    mx = maximum_filter(
        nu,
        size=3,
        mode="constant",
        cval=-np.inf,
    )

    is_peak = (
        valid
        &
        (nu == mx)
        &
        (nu >= threshold)
    )

    coords = np.argwhere(
        is_peak
    )

    if len(
        coords
    ) == 0:

        return (
            np.empty(
                (
                    0,
                    2,
                ),
                dtype=np.int32,
            ),

            np.empty(
                (0,),
                dtype=np.float64,
            ),
        )

    vals = nu[
        coords[:, 0],
        coords[:, 1],
    ]

    return (
        coords.astype(
            np.int32
        ),

        vals.astype(
            np.float64
        ),
    )


def match_catalogs(
    truth_coords,
    rec_coords,
    radius_pix,
):
    if (
        len(truth_coords)
        == 0
        or
        len(rec_coords)
        == 0
    ):
        return []

    tree = cKDTree(
        rec_coords
    )

    candidates = []

    for it, coord in enumerate(
        truth_coords
    ):

        inds = tree.query_ball_point(
            coord,
            radius_pix,
        )

        for ir in inds:

            d = float(
                np.linalg.norm(
                    coord
                    - rec_coords[
                        ir
                    ]
                )
            )

            candidates.append(
                (
                    d,
                    it,
                    ir,
                )
            )

    candidates.sort(
        key=lambda x: x[0]
    )

    used_t = set()
    used_r = set()

    matches = []

    for d, it, ir in candidates:

        if (
            it in used_t
            or
            ir in used_r
        ):
            continue

        used_t.add(it)
        used_r.add(ir)

        matches.append(
            (
                it,
                ir,
                d,
            )
        )

    return matches


def local_max_value(
    image,
    coord,
    valid,
    radius_pix,
):
    y0, x0 = coord

    rr = int(
        np.ceil(
            radius_pix
        )
    )

    y1 = max(
        0,
        y0 - rr,
    )

    y2 = min(
        N,
        y0 + rr + 1,
    )

    x1 = max(
        0,
        x0 - rr,
    )

    x2 = min(
        N,
        x0 + rr + 1,
    )

    yy, xx = np.mgrid[
        y1:y2,
        x1:x2
    ]

    dist = np.sqrt(
        (yy - y0) ** 2
        + (xx - x0) ** 2
    )

    ok = (
        valid[
            y1:y2,
            x1:x2
        ]
        &
        (
            dist
            <= radius_pix
        )
    )

    if not np.any(
        ok
    ):
        return np.nan

    return float(
        np.max(
            image[
                y1:y2,
                x1:x2
            ][ok]
        )
    )


# =============================================================================
# PEAK ANALYSIS
# =============================================================================
print()
print("=" * 120)
print("PEAK ANALYSIS")
print("=" * 120)

pr_rows = []
height_rows = []

for smooth_arcmin in (
    SMOOTH_ARCMIN
):

    sigma_pix = (
        smooth_arcmin
        / PIXEL_ARCMIN
    )

    print()
    print(
        f"Smoothing = "
        f"{smooth_arcmin:g} arcmin"
    )

    for i, (
        c,
        r,
        p,
    ) in enumerate(meta):

        (
            truth_sm,
            truth_nu,
            truth_valid,
        ) = make_nu_map(
            truths[i],
            masks[i],
            sigma_pix,
        )

        rec_cache = {}

        for method in METHODS:

            (
                rec_sm,
                rec_nu,
                rec_valid,
            ) = make_nu_map(
                maps[
                    method
                ][i],
                masks[i],
                sigma_pix,
            )

            rec_cache[
                method
            ] = (
                rec_sm,
                rec_nu,
                (
                    truth_valid
                    &
                    rec_valid
                ),
            )

        for truth_thr in (
            TRUTH_NU_THRESHOLDS
        ):

            (
                truth_coords,
                truth_vals,
            ) = peak_catalog(
                truth_nu,
                truth_valid,
                truth_thr,
            )

            # -------------------------------------------------
            # PEAK HEIGHT
            # -------------------------------------------------
            for method in METHODS:

                (
                    rec_sm,
                    rec_nu,
                    valid,
                ) = rec_cache[
                    method
                ]

                for ip, coord in enumerate(
                    truth_coords
                ):

                    y0, x0 = coord

                    truth_amp = float(
                        truth_sm[
                            y0,
                            x0,
                        ]
                    )

                    rec_amp = (
                        local_max_value(
                            rec_sm,
                            coord,
                            valid,
                            MATCH_RADIUS_PIX,
                        )
                    )

                    if abs(
                        truth_amp
                    ) > 1e-15:

                        ratio = (
                            rec_amp
                            / truth_amp
                        )

                    else:
                        ratio = np.nan

                    height_rows.append(
                        {
                            "method": method,

                            "c": c,
                            "r": r,
                            "p": p,

                            "smooth_arcmin": (
                                smooth_arcmin
                            ),

                            "truth_nu_threshold": (
                                truth_thr
                            ),

                            "truth_nu": float(
                                truth_vals[ip]
                            ),

                            "truth_amp": (
                                truth_amp
                            ),

                            "rec_amp": (
                                rec_amp
                            ),

                            "local_ratio": (
                                ratio
                            ),
                        }
                    )

            # -------------------------------------------------
            # PRECISION / RECALL CURVES
            # -------------------------------------------------
            for method in METHODS:

                (
                    rec_sm,
                    rec_nu,
                    valid,
                ) = rec_cache[
                    method
                ]

                if len(
                    truth_coords
                ):

                    keep = valid[
                        truth_coords[
                            :,
                            0,
                        ],
                        truth_coords[
                            :,
                            1,
                        ],
                    ]

                    tc = truth_coords[
                        keep
                    ]

                else:
                    tc = truth_coords

                for rec_thr in (
                    REC_NU_THRESHOLDS
                ):

                    (
                        rec_coords,
                        _,
                    ) = peak_catalog(
                        rec_nu,
                        valid,
                        rec_thr,
                    )

                    matches = (
                        match_catalogs(
                            tc,
                            rec_coords,
                            MATCH_RADIUS_PIX,
                        )
                    )

                    pr_rows.append(
                        {
                            "method": method,

                            "c": c,
                            "r": r,
                            "p": p,

                            "smooth_arcmin": (
                                smooth_arcmin
                            ),

                            "truth_nu_threshold": (
                                truth_thr
                            ),

                            "rec_nu_threshold": (
                                float(
                                    rec_thr
                                )
                            ),

                            "n_truth": int(
                                len(tc)
                            ),

                            "n_rec": int(
                                len(
                                    rec_coords
                                )
                            ),

                            "n_match": int(
                                len(
                                    matches
                                )
                            ),
                        }
                    )

        if (
            (i + 1)
            % 80
            == 0
        ):

            print(
                f"  {i+1:3d}/240",
                flush=True,
            )


# =============================================================================
# SAVE PEAK RAW RESULTS
# =============================================================================
pr = pd.DataFrame(
    pr_rows
)

height = pd.DataFrame(
    height_rows
)

pr.to_csv(
    OUT
    / "peak_PR_all.csv",
    index=False,
)

height.to_csv(
    OUT
    / "peak_height_all.csv",
    index=False,
)


# =============================================================================
# AGGREGATE PR
# =============================================================================
pr_summary = (
    pr
    .groupby(
        [
            "method",
            "smooth_arcmin",
            "truth_nu_threshold",
            "rec_nu_threshold",
        ]
    )
    .agg(
        n_truth=(
            "n_truth",
            "sum",
        ),

        n_rec=(
            "n_rec",
            "sum",
        ),

        n_match=(
            "n_match",
            "sum",
        ),
    )
    .reset_index()
)

pr_summary[
    "completeness"
] = (
    pr_summary.n_match
    / pr_summary.n_truth.replace(
        0,
        np.nan,
    )
)

pr_summary[
    "purity"
] = (
    pr_summary.n_match
    / pr_summary.n_rec.replace(
        0,
        np.nan,
    )
)

pr_summary.to_csv(
    OUT
    / "peak_PR_summary.csv",
    index=False,
)


# =============================================================================
# HEIGHT SUMMARY
# =============================================================================
height_summary = (
    height
    .groupby(
        [
            "method",
            "smooth_arcmin",
            "truth_nu_threshold",
        ]
    )
    .agg(
        n=(
            "local_ratio",
            "size",
        ),

        local_ratio_median=(
            "local_ratio",
            "median",
        ),

        local_ratio_mean=(
            "local_ratio",
            "mean",
        ),
    )
    .reset_index()
)

height_summary.to_csv(
    OUT
    / "peak_height_summary.csv",
    index=False,
)


# =============================================================================
# PREPARE PR CURVE
# =============================================================================
def prepare_pr_curve(
    q,
):
    q = q[
        np.isfinite(
            q.completeness
        )
        &
        np.isfinite(
            q.purity
        )
    ].copy()

    if len(q) < 2:
        return None

    q = (
        q
        .groupby(
            "completeness",
            as_index=False,
        )["purity"]
        .max()
        .sort_values(
            "completeness"
        )
    )

    return (
        q.completeness.to_numpy(),
        q.purity.to_numpy(),
    )


# =============================================================================
# COMMON RANGE AUC
# =============================================================================
auc_rows = []

for smooth in (
    SMOOTH_ARCMIN
):

    for truth_thr in (
        TRUTH_NU_THRESHOLDS
    ):

        curves = {}

        for method in METHODS:

            q = pr_summary[
                (
                    pr_summary.method
                    == method
                )
                &
                (
                    pr_summary.smooth_arcmin
                    == smooth
                )
                &
                (
                    pr_summary.truth_nu_threshold
                    == truth_thr
                )
            ]

            curves[
                method
            ] = prepare_pr_curve(
                q
            )

        valid_curves = [
            x
            for x in curves.values()
            if x is not None
        ]

        cmin = max(
            np.min(c)
            for c, _
            in valid_curves
        )

        cmax = min(
            np.max(c)
            for c, _
            in valid_curves
        )

        if cmax <= cmin:
            continue

        grid = np.linspace(
            cmin,
            cmax,
            500,
        )

        for method in METHODS:

            curve = curves[
                method
            ]

            if curve is None:
                continue

            c, purity = curve

            yi = np.interp(
                grid,
                c,
                purity,
            )

            auc = (
                np.trapz(
                    yi,
                    grid,
                )
                / (
                    cmax
                    - cmin
                )
            )

            auc_rows.append(
                {
                    "method": method,

                    "smooth_arcmin": (
                        smooth
                    ),

                    "truth_nu_threshold": (
                        truth_thr
                    ),

                    "common_cmin": (
                        cmin
                    ),

                    "common_cmax": (
                        cmax
                    ),

                    "normalized_common_auc": (
                        auc
                    ),
                }
            )

auc = pd.DataFrame(
    auc_rows
)

auc.to_csv(
    OUT
    / "peak_PR_AUC.csv",
    index=False,
)


# =============================================================================
# FINAL SCIENCE SUMMARY
# =============================================================================
science_rows = []

for method in METHODS:

    f = field_summary[
        field_summary.method
        == method
    ].iloc[0]

    s = spectral_score[
        spectral_score.method
        == method
    ].iloc[0]

    row = {
        "method": method,

        "nmse": float(
            f.nmse_mean
        ),

        "pcc": float(
            f.pcc_mean
        ),

        "std_ratio": float(
            f.std_ratio_mean
        ),

        "mean_abs_T_error": float(
            s.mean_abs_T_error
        ),

        "mean_abs_A_error": float(
            s.mean_abs_A_error
        ),

        "mean_r_ell": float(
            s.mean_r_ell
        ),
    }

    # ---------------------------------------------------------
    # use nu_true >= 2.5 for compact PR comparison
    # ---------------------------------------------------------
    for smooth in [
        2.0,
        4.0,
        6.0,
    ]:

        q = auc[
            (
                auc.method
                == method
            )
            &
            (
                auc.smooth_arcmin
                == smooth
            )
            &
            (
                auc.truth_nu_threshold
                == 2.5
            )
        ]

        row[
            f"PR_AUC_{int(smooth)}arc"
        ] = (
            float(
                q.normalized_common_auc.iloc[
                    0
                ]
            )
            if len(q)
            else np.nan
        )

    # ---------------------------------------------------------
    # use nu_true >= 3 for amplitude comparison
    # ---------------------------------------------------------
    for smooth in [
        2.0,
        4.0,
        6.0,
    ]:

        q = height_summary[
            (
                height_summary.method
                == method
            )
            &
            (
                height_summary.smooth_arcmin
                == smooth
            )
            &
            (
                height_summary.truth_nu_threshold
                == 3.0
            )
        ]

        row[
            f"peak_amp_{int(smooth)}arc"
        ] = (
            float(
                q.local_ratio_median.iloc[
                    0
                ]
            )
            if len(q)
            else np.nan
        )

    # ---------------------------------------------------------
    # candidate weights
    # ---------------------------------------------------------
    match = [
        x
        for x in CANDIDATES
        if x[0] == method
    ]

    if match:

        _, a1, a2 = (
            match[0]
        )

        row[
            "a1"
        ] = a1

        row[
            "a2"
        ] = a2

    else:

        row["a1"] = np.nan
        row["a2"] = np.nan

    science_rows.append(
        row
    )


science = pd.DataFrame(
    science_rows
)

science.to_csv(
    OUT
    / "SCIENCE_SUMMARY.csv",
    index=False,
)


# =============================================================================
# PRINT FINAL RESULTS
# =============================================================================
print()
print("=" * 185)
print("SCIENCE SUMMARY")
print("=" * 185)

print(
    science.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.5f}"
        ),
    )
)


# =============================================================================
# CANDIDATES ONLY
# =============================================================================
candidate_science = science[
    science.method.str.startswith(
        "W_A1_"
    )
].copy()

candidate_science = (
    candidate_science
    .sort_values(
        "nmse"
    )
    .reset_index(
        drop=True
    )
)

print()
print("=" * 185)
print("CANDIDATES SORTED BY NMSE")
print("=" * 185)

print(
    candidate_science.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.5f}"
        ),
    )
)


# =============================================================================
# REFERENCE H4
# =============================================================================
H4_NAME = (
    candidate_name(
        1.0,
        1.0,
    )
)

print()
print("=" * 120)
print("H4 REFERENCE")
print("=" * 120)

print(
    "H4 corresponds exactly to:"
)

print(
    f"  {H4_NAME}"
)

print(
    "  a0=0, a1=1, a2=1, a3=0, "
    "coarse=G+S"
)


# =============================================================================
# OUTPUT
# =============================================================================
print()
print("=" * 120)
print("FILES")
print("=" * 120)

for fn in [
    "field_metrics_all.csv",
    "field_metrics_summary.csv",
    "field_by_cosmology.csv",
    "field_bootstrap.csv",
    "spectral_all.csv",
    "spectral_summary.csv",
    "spectral_scores.csv",
    "peak_PR_all.csv",
    "peak_PR_summary.csv",
    "peak_PR_AUC.csv",
    "peak_height_all.csv",
    "peak_height_summary.csv",
    "SCIENCE_SUMMARY.csv",
]:
    print(
        OUT / fn
    )

print()
print(
    "IMPORTANT:"
)

print(
    "Only validation R0-R9 was used."
)

print(
    "Do not evaluate R10-R29 until one "
    "single band-weight combination is frozen."
)

print()
print(
    "WEIGHTED BAND FUSION VALIDATION COMPLETE"
)
