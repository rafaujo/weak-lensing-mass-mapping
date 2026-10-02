from project_paths import PATHS
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree


# =============================================================================
# FROZEN TEST CONFIGURATION
# =============================================================================
N = 176
J = 4
PIXEL_ARCMIN = 2.0

A0 = 0.0
A1 = 0.75
A2 = 0.75
A3 = 0.0

METHOD_HYBRID = "W_A1_075_A2_075"

BASE = Path(
    str(PATHS.work_root / 'test/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

GS_FILE = Path(
    str(PATHS.work_root / 'test/starlet_gaussian_prior/gaussian_starlet_test_reconstruction.npy')
)

MC_DIR = Path(
    str(PATHS.work_root / 'test/mcalens/recon')
)

OUT = Path(
    str(PATHS.work_root / 'test/weighted_band_fusion')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

HYBRID_FILE = (
    OUT
    / "W_A1_075_A2_075_test_maps.npy"
)

FIELD_ALL_FILE = (
    OUT
    / "field_metrics_all.csv"
)

FIELD_SUMMARY_FILE = (
    OUT
    / "field_metrics_summary.csv"
)

FIELD_BYCOSMO_FILE = (
    OUT
    / "field_metrics_by_cosmology.csv"
)

BOOT_FILE = (
    OUT
    / "field_bootstrap.csv"
)

SPECTRAL_ALL_FILE = (
    OUT
    / "spectral_all.csv"
)

SPECTRAL_SUMMARY_FILE = (
    OUT
    / "spectral_summary.csv"
)

SPECTRAL_SCORE_FILE = (
    OUT
    / "spectral_scores.csv"
)

PEAK_PR_ALL_FILE = (
    OUT
    / "peak_PR_all.csv"
)

PEAK_PR_SUMMARY_FILE = (
    OUT
    / "peak_PR_summary.csv"
)

PEAK_AUC_FILE = (
    OUT
    / "peak_PR_AUC.csv"
)

PEAK_HEIGHT_ALL_FILE = (
    OUT
    / "peak_height_all.csv"
)

PEAK_HEIGHT_SUMMARY_FILE = (
    OUT
    / "peak_height_summary.csv"
)

SCIENCE_FILE = (
    OUT
    / "SCIENCE_SUMMARY.csv"
)


# =============================================================================
# PATCHES
# =============================================================================
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
    20,
    8,
    N,
    N,
)

FLAT_SHAPE = (
    480,
    N,
    N,
)


# =============================================================================
# FROZEN TEST REFERENCES
# =============================================================================
EXPECTED_GS_NMSE = 0.642805480
EXPECTED_GS_PCC = 0.59660041

EXPECTED_MC_NMSE = 0.651369139
EXPECTED_MC_PCC = 0.595698

SANITY_TOL = 2.0e-3


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

METHODS = [
    "GAUSSIAN_STARLET",
    "MCALENS",
    METHOD_HYBRID,
]


# =============================================================================
# BOOTSTRAP
# =============================================================================
NBOOT = 30000
BOOT_SEED = 12345


# =============================================================================
# BASIC METRICS
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

    # Mass-sheet invariant.
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
                closure - 1.0
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
# LOAD TEST TRUTH + MASK
# =============================================================================
print()
print("=" * 120)
print(
    "FROZEN WEIGHTED-BAND FUSION — HELD-OUT TEST"
)
print("=" * 120)

print(
    "a0/a1/a2/a3 :",
    A0,
    A1,
    A2,
    A3,
)

print(
    "coarse       : Gaussian+Starlet"
)

print(
    "Starlet closure:",
    CLOSURE_ERROR,
)

if CLOSURE_ERROR > 1e-12:
    raise RuntimeError(
        "Starlet filter closure failure."
    )


mask_full = np.load(
    MASK_FILE
).astype(bool)

truths = []
masks = []
meta = []

for c in range(3):

    for r in range(
        10,
        30,
    ):

        fn = (
            BASE
            / f"C{c}_R{r}.npz"
        )

        if not fn.exists():
            raise FileNotFoundError(
                fn
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

print(
    "truths:",
    truths.shape,
)


# =============================================================================
# LOAD GAUSSIAN + STARLET TEST MAPS
# =============================================================================
if not GS_FILE.exists():
    raise FileNotFoundError(
        GS_FILE
    )

gs_grid = np.load(
    GS_FILE,
    mmap_mode="r",
)

if gs_grid.shape != GRID_SHAPE:
    raise RuntimeError(
        f"Unexpected G+S shape: "
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
        r - 10,
        p,
    ]

del gs_grid


# =============================================================================
# LOAD MCALENS TEST MAPS
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

        if "mcalens" not in z:
            raise RuntimeError(
                f"{fn} has no 'mcalens' key"
            )

        mc[i] = np.asarray(
            z["mcalens"],
            dtype=np.float32,
        )

print(
    "G+S     :",
    gs.shape,
)

print(
    "MCALens :",
    mc.shape,
)


# =============================================================================
# BASELINE SANITY
# =============================================================================
def summarize_maps(
    name,
    arr,
):
    vals = []

    for i in range(
        480
    ):
        vals.append(
            ms_metrics(
                arr[i],
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

if (
    abs(
        gs_check["nmse"]
        - EXPECTED_GS_NMSE
    )
    > SANITY_TOL
):

    raise RuntimeError(
        "G+S test mismatch."
    )

if (
    abs(
        mc_check["nmse"]
        - EXPECTED_MC_NMSE
    )
    > SANITY_TOL
):

    raise RuntimeError(
        "MCALens test mismatch."
    )


# =============================================================================
# BUILD FROZEN HYBRID
# =============================================================================
print()
print("=" * 120)
print("BUILDING FROZEN HYBRID")
print("=" * 120)

hybrid = np.empty(
    FLAT_SHAPE,
    dtype=np.float32,
)

BATCH = 16

max_err_gs = 0.0
max_err_mc = 0.0

for ib in range(
    0,
    480,
    BATCH,
):

    ie = min(
        ib + BATCH,
        480,
    )

    gs_d, gs_c = (
        starlet_decompose_batch(
            gs[
                ib:ie
            ]
        )
    )

    mc_d, mc_c = (
        starlet_decompose_batch(
            mc[
                ib:ie
            ]
        )
    )

    # Verify exact reconstruction.
    gs_back = (
        gs_c
        + np.sum(
            gs_d,
            axis=1,
        )
    )

    mc_back = (
        mc_c
        + np.sum(
            mc_d,
            axis=1,
        )
    )

    max_err_gs = max(
        max_err_gs,
        float(
            np.max(
                np.abs(
                    gs_back
                    - gs[
                        ib:ie
                    ]
                )
            )
        ),
    )

    max_err_mc = max(
        max_err_mc,
        float(
            np.max(
                np.abs(
                    mc_back
                    - mc[
                        ib:ie
                    ]
                )
            )
        ),
    )

    # ---------------------------------------------------------
    # Frozen fusion:
    #
    # j0 = 100% G+S
    # j1 = 25% G+S + 75% MCALens
    # j2 = 25% G+S + 75% MCALens
    # j3 = 100% G+S
    # coarse = 100% G+S
    # ---------------------------------------------------------
    h = (
        gs_c

        + gs_d[
            :,
            0,
        ]

        + (
            (
                1.0
                - A1
            )
            * gs_d[
                :,
                1,
            ]
            +
            A1
            * mc_d[
                :,
                1,
            ]
        )

        + (
            (
                1.0
                - A2
            )
            * gs_d[
                :,
                2,
            ]
            +
            A2
            * mc_d[
                :,
                2,
            ]
        )

        + gs_d[
            :,
            3,
        ]
    )

    hybrid[
        ib:ie
    ] = h.astype(
        np.float32
    )

    print(
        f"{ie:3d}/480",
        flush=True,
    )

print()
print(
    "max G+S reconstruction error:",
    f"{max_err_gs:.3e}",
)

print(
    "max MC reconstruction error :",
    f"{max_err_mc:.3e}",
)

np.save(
    HYBRID_FILE,
    hybrid,
)

maps = {
    "GAUSSIAN_STARLET": gs,
    "MCALENS": mc,
    METHOD_HYBRID: hybrid,
}


# =============================================================================
# FIELD METRICS
# =============================================================================
field_rows = []

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
            std_ratio,
        ) = ms_metrics(
            arr[i],
            truths[i],
            masks[i],
        )

        field_rows.append(
            {
                "method": method,

                "c": c,
                "r": r,
                "p": p,

                "mse": mse,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": std_ratio,
            }
        )

field = pd.DataFrame(
    field_rows
)

field.to_csv(
    FIELD_ALL_FILE,
    index=False,
)

field_summary = (
    field
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

        mse_mean=(
            "mse",
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
    .sort_values(
        "nmse_mean"
    )
)

field_summary.to_csv(
    FIELD_SUMMARY_FILE,
    index=False,
)

print()
print("=" * 135)
print("FIELD-LEVEL HELD-OUT TEST")
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
byc = (
    field
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

byc.to_csv(
    FIELD_BYCOSMO_FILE,
    index=False,
)

print()
print("=" * 120)
print("BY COSMOLOGY")
print("=" * 120)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =============================================================================
# REALIZATION-BLOCK BOOTSTRAP
# =============================================================================
def paired_block_bootstrap(
    candidate,
    baseline,
):
    a = field[
        field.method
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

    b = field[
        field.method
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

    if len(q) != 480:
        raise RuntimeError(
            f"Matched cases: {len(q)}"
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
        BOOT_SEED
    )

    inds = rng.integers(
        0,
        len(block),
        size=(
            NBOOT,
            len(block),
        ),
    )

    base_boot = (
        nb[
            inds
        ].mean(
            axis=1
        )
    )

    cand_boot = (
        na[
            inds
        ].mean(
            axis=1
        )
    )

    nmse_gain_boot = (
        100.0
        * (
            base_boot
            - cand_boot
        )
        / base_boot
    )

    pcc_delta_boot = (
        pa[
            inds
        ].mean(
            axis=1
        )
        -
        pb[
            inds
        ].mean(
            axis=1
        )
    )

    base_nmse = float(
        nb.mean()
    )

    cand_nmse = float(
        na.mean()
    )

    nmse_gain = (
        100.0
        * (
            base_nmse
            - cand_nmse
        )
        / base_nmse
    )

    pcc_delta = float(
        pa.mean()
        - pb.mean()
    )

    nlo, nhi = np.percentile(
        nmse_gain_boot,
        [
            2.5,
            97.5,
        ],
    )

    plo, phi = np.percentile(
        pcc_delta_boot,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "candidate": candidate,
        "baseline": baseline,

        "candidate_nmse": cand_nmse,
        "baseline_nmse": base_nmse,

        "nmse_gain_pct": (
            nmse_gain
        ),

        "nmse_gain_ci_lo": float(
            nlo
        ),

        "nmse_gain_ci_hi": float(
            nhi
        ),

        "pcc_delta": (
            pcc_delta
        ),

        "pcc_ci_lo": float(
            plo
        ),

        "pcc_ci_hi": float(
            phi
        ),

        "patch_wins_nmse": int(
            np.sum(
                q.nmse_a
                <
                q.nmse_b
            )
        ),

        "patch_total": 480,

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

for baseline in [
    "GAUSSIAN_STARLET",
    "MCALENS",
]:

    boot_rows.append(
        paired_block_bootstrap(
            METHOD_HYBRID,
            baseline,
        )
    )

boot = pd.DataFrame(
    boot_rows
)

boot.to_csv(
    BOOT_FILE,
    index=False,
)

print()
print("=" * 155)
print("PAIRED REALIZATION-BLOCK BOOTSTRAP")
print("=" * 155)

print(
    boot.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
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

    Ptt = (
        np.abs(T) ** 2
    )

    Prr = (
        np.abs(R) ** 2
    )

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

        rows.append(
            {
                "bin": b,

                "ell": float(
                    np.mean(
                        ELL[sel]
                    )
                ),

                "T_cross": (
                    rt / tt
                ),

                "A_auto": (
                    rr / tt
                ),

                "r_ell": (
                    rt
                    / np.sqrt(
                        max(
                            rr * tt,
                            1e-30,
                        )
                    )
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
            f"{i+1:3d}/480",
            flush=True,
        )

spec = pd.DataFrame(
    spec_rows
)

spec.to_csv(
    SPECTRAL_ALL_FILE,
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
    SPECTRAL_SUMMARY_FILE,
    index=False,
)

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
    SPECTRAL_SCORE_FILE,
    index=False,
)

print()
print(
    spectral_score.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
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
        denominator > 1e-12
    )

    smoothed[
        good
    ] = (
        numerator[
            good
        ]
        / denominator[
            good
        ]
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
            "No valid pixels after smoothing."
        )

    sigma_map = float(
        np.std(
            sm[
                valid
            ]
        )
    )

    if sigma_map <= 0:
        raise RuntimeError(
            "Invalid map sigma."
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

    if len(coords) == 0:

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

        used_t.add(
            it
        )

        used_r.add(
            ir
        )

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
            # PEAK HEIGHT RESPONSE
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
            # PRECISION / RECALL
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
                f"  {i+1:3d}/480",
                flush=True,
            )


# =============================================================================
# SAVE PEAK RAW TABLES
# =============================================================================
pr = pd.DataFrame(
    pr_rows
)

height = pd.DataFrame(
    height_rows
)

pr.to_csv(
    PEAK_PR_ALL_FILE,
    index=False,
)

height.to_csv(
    PEAK_HEIGHT_ALL_FILE,
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
    PEAK_PR_SUMMARY_FILE,
    index=False,
)


# =============================================================================
# PEAK HEIGHT SUMMARY
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
    PEAK_HEIGHT_SUMMARY_FILE,
    index=False,
)


# =============================================================================
# PR CURVE HELPER
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
# COMMON-RANGE PR AUC
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
    PEAK_AUC_FILE,
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

    # PR-AUC at truth nu >= 2.5
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

    # Peak amplitude at truth nu >= 3.
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

    science_rows.append(
        row
    )

science = pd.DataFrame(
    science_rows
)

science.to_csv(
    SCIENCE_FILE,
    index=False,
)

print()
print("=" * 180)
print("SCIENCE SUMMARY — HELD-OUT TEST")
print("=" * 180)

print(
    science.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.5f}"
        ),
    )
)


# =============================================================================
# VALIDATION -> TEST GENERALIZATION
# =============================================================================
VALIDATION_REFERENCE = {
    "nmse": 0.63953223,
    "pcc": 0.59959907,
    "std_ratio": 0.63627692,
    "PR_AUC_2arc": 0.62816,
    "PR_AUC_4arc": 0.61534,
    "PR_AUC_6arc": 0.63460,
    "peak_amp_2arc": 0.58057,
    "peak_amp_4arc": 0.82913,
    "peak_amp_6arc": 0.87231,
}

hyb = science[
    science.method
    == METHOD_HYBRID
].iloc[0]

print()
print("=" * 120)
print("VALIDATION -> TEST GENERALIZATION")
print("=" * 120)

for key, val_value in (
    VALIDATION_REFERENCE.items()
):

    test_value = float(
        hyb[key]
    )

    print(
        f"{key:20s} "
        f"validation={val_value:.6f}  "
        f"test={test_value:.6f}  "
        f"delta={test_value-val_value:+.6f}"
    )


# =============================================================================
# OUTPUT
# =============================================================================
print()
print("=" * 120)
print("OUTPUT FILES")
print("=" * 120)

for fn in [
    HYBRID_FILE,
    FIELD_ALL_FILE,
    FIELD_SUMMARY_FILE,
    FIELD_BYCOSMO_FILE,
    BOOT_FILE,
    SPECTRAL_ALL_FILE,
    SPECTRAL_SUMMARY_FILE,
    SPECTRAL_SCORE_FILE,
    PEAK_PR_ALL_FILE,
    PEAK_PR_SUMMARY_FILE,
    PEAK_AUC_FILE,
    PEAK_HEIGHT_ALL_FILE,
    PEAK_HEIGHT_SUMMARY_FILE,
    SCIENCE_FILE,
]:
    print(
        fn
    )

print()
print(
    "FROZEN TEST COMPLETE."
)

print(
    "No further tuning of a1/a2 should be performed "
    "using R10-R29."
)
