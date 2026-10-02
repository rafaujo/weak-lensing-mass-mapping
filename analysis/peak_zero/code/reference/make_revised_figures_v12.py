#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_revised_figures_v5.py
==========================

Publication figures for the revised weak-lensing manuscript.

READ-ONLY POST-PROCESSING PIPELINE.

This script DOES NOT:
    - reconstruct maps
    - tune hyperparameters
    - modify lambda / beta
    - modify MCALens Nsigma
    - modify multiscale-fusion weights
    - use held-out truth for tuning

Frozen held-out test:
    cosmologies     : C0, C1, C2
    realizations    : R10--R29
    patches         : P0--P7
    total subfields : 3 x 20 x 8 = 480

Main science methods:
    Gaussian+Starlet
    MCALens
    frozen multiscale fusion:
        a0 = 0
        a1 = 0.75
        a2 = 0.75
        a3 = 0
        coarse = Gaussian+Starlet

Figures:
    fig_maps_revised
    fig_field_metrics_revised
    fig_ablation_revised
    fig_spectral_revised
    fig_peak_pr_revised
    fig_peak_amplitude_revised
"""
from project_paths import PATHS

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


# =============================================================================
# V8 ONLY — PRECOMPUTED WIENER / GAUSSIAN-ONLY MAPS
#
# Figure generation only. No Wiener reconstruction is run here.
# =============================================================================

V8_WIENER_FILE = Path(
    str(PATHS.work_root / 'wiener_gaussian_only/wiener_test_reconstruction.npy')
)


def v8_load_wiener():

    arr = np.asarray(
        np.load(
            V8_WIENER_FILE,
            mmap_mode="r",
        )
    )

    print()
    print("=" * 100)
    print("V8 WIENER MAPS")
    print("=" * 100)
    print("file :", V8_WIENER_FILE)
    print("shape:", arr.shape)

    if arr.shape == (3, 20, 8, 176, 176):
        return arr

    if arr.shape == (3, 30, 8, 176, 176):
        print("Using test realisations [10:30].")
        return arr[:, 10:30]

    if arr.shape == (480, 176, 176):
        return arr.reshape(
            3,
            20,
            8,
            176,
            176,
        )

    raise RuntimeError(
        f"Unexpected Wiener array shape: {arr.shape}"
    )


V8_WIENER_MAPS = v8_load_wiener()




# =============================================================================
# CONFIGURATION
# =============================================================================

N = 176

PIX_ARCMIN = 2.0

TEST_REALIZATIONS = list(
    range(
        10,
        30,
    )
)

N_TEST = len(
    TEST_REALIZATIONS
)

N_COSMO = 3
N_PATCH = 8


# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = Path(
    str(PATHS.data_root)
)

KAPPA_FILE = (
    DATA_DIR
    / "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
)

MASK_FILE = (
    DATA_DIR
    / "WIDE12H_bin2_2arcmin_mask.npy"
)

LABEL_FILE = (
    DATA_DIR
    / "sampled_label.npy"
)


GS_FILE = Path(
    str(PATHS.work_root / 'test/starlet_gaussian_prior/gaussian_starlet_test_reconstruction.npy')
)

MC_ROOT = Path(
    str(PATHS.work_root / 'test/mcalens/recon')
)

FUSION_FILE = Path(
    str(PATHS.work_root / 'test/weighted_band_fusion/W_A1_075_A2_075_test_maps.npy')
)


OUT = Path(
    str(PATHS.work_root / 'test/revised_paper_figures_v12')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# MCALENS KEY
# =============================================================================

MC_KEY = None


# =============================================================================
# FIXED VISUAL EXAMPLE
# =============================================================================

VIS_C = 0
VIS_R = 10
VIS_P = 3


# =============================================================================
# PEAK DEFINITIONS
# =============================================================================

SMOOTH_ARCMIN = [
    2.0,
    4.0,
    6.0,
]

TRUTH_NU_PR = 2.5

TRUTH_NU_AMP = 3.0

MATCH_RADIUS_ARCMIN = 4.0

MATCH_RADIUS_PIX = (
    MATCH_RADIUS_ARCMIN
    / PIX_ARCMIN
)

MASK_SUPPORT_MIN = 0.70


REC_NU_THRESHOLDS = np.linspace(
    8.0,
    -0.5,
    120,
)


# =============================================================================
# SPECTRAL DEFINITIONS
# =============================================================================

N_ELL_BINS = 16

SPECTRAL_AMPLITUDE_YMIN = 0.0
SPECTRAL_AMPLITUDE_YMAX = 1.10

SPECTRAL_CORR_YMIN = 0.2
SPECTRAL_CORR_YMAX = 1.03


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


C = {
    "KS": "#7f8c8d",
    "GLIMPSE": "#999999",
        "WIENER": "#CC79A7",
"STARLET_NOMASK": "#9ecae1",
    "STARLET_MASK": "#3182bd",
    "GAUSSIAN_STARLET": "#0072B2",
    "MCALENS": "#D55E00",
    "FUSION": "#009E73",
}


LABEL = {
    "KS": "Kaiser–Squires",
    "GLIMPSE": "GLIMPSE",
        "WIENER": "Wiener-like",
"STARLET_NOMASK": "Starlet\nno mask",
    "STARLET_MASK": "Starlet\nmask",
    "GAUSSIAN_STARLET": "Gaussian\n+Starlet",
    "MCALENS": "MCALens",
    "FUSION": "Multiscale\nfusion",
}


# =============================================================================
# FROZEN HELD-OUT FIELD RESULTS
# =============================================================================

FIELD = pd.DataFrame([

    {
        "method": "KS",
        "nmse": 0.724242146,
        "pcc": 0.531082,
        "std": 0.606745,
    },

    {
        "method": "GLIMPSE",
        "nmse": 0.824143,
        "pcc": 0.437311,
        "std": 0.564743,
    },

        {
        "method": "WIENER",
        "nmse": 0.686769155,
        "pcc": 0.568533165,
        "std": 0.471536857,
    },
{
        "method": "STARLET_NOMASK",
        "nmse": 0.693355,
        "pcc": 0.555284,
        "std": 0.487017,
    },

    {
        "method": "STARLET_MASK",
        "nmse": 0.678849,
        "pcc": 0.564984,
        "std": 0.531114,
    },

    {
        "method": "MCALENS",
        "nmse": 0.651369139,
        "pcc": 0.595698,
        "std": 0.685784,
    },

    {
        "method": "GAUSSIAN_STARLET",
        "nmse": 0.642805480,
        "pcc": 0.59660041,
        "std": 0.58345016,
    },

    {
        "method": "FUSION",
        "nmse": 0.634196,
        "pcc": 0.603865,
        "std": 0.631480,
    },
])


EXPECTED_MAP_METRICS = {

    "GAUSSIAN_STARLET": (
        0.642805480,
        0.59660041,
        0.58345016,
    ),

    "MCALENS": (
        0.651369139,
        0.595698,
        0.685784,
    ),

    "FUSION": (
        0.634196,
        0.603865,
        0.631480,
    ),
}


# =============================================================================
# FILE HELPERS
# =============================================================================

def require(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            path
        )


for fn in [
    KAPPA_FILE,
    MASK_FILE,
    LABEL_FILE,
    GS_FILE,
    FUSION_FILE,
]:

    require(
        fn
    )


# =============================================================================
# LOAD FAIR
# =============================================================================

kappa_vec = np.load(
    KAPPA_FILE,
    mmap_mode="r",
)

mask_full = np.load(
    MASK_FILE
).astype(bool)

labels = np.load(
    LABEL_FILE,
    mmap_mode="r",
)


print()
print("=" * 120)
print("FINAL FIGURE PIPELINE — REVISED PAPER V5")
print("=" * 120)

print(
    "kappa shape :",
    kappa_vec.shape,
)

print(
    "mask shape  :",
    mask_full.shape,
)

print(
    "f_obs full  :",
    mask_full.mean(),
)

print(
    "label shape :",
    labels.shape,
)


# =============================================================================
# PATCH GEOMETRY
# =============================================================================

PATCH_ROWS = [
    (
        p * N,
        (p + 1) * N,
    )
    for p in range(
        N_PATCH
    )
]


MASK_PATCH = np.stack(
    [
        mask_full[
            r0:r1,
            :
        ]
        for (
            r0,
            r1,
        ) in PATCH_ROWS
    ],
    axis=0,
)


# =============================================================================
# TRUTH
# =============================================================================

def vector_to_full(vec):

    out = np.zeros(
        mask_full.shape,
        dtype=np.float32,
    )

    out[
        mask_full
    ] = np.asarray(
        vec,
        dtype=np.float32,
    )

    return out


def build_truth_grid():

    truth = np.empty(
        (
            N_COSMO,
            N_TEST,
            N_PATCH,
            N,
            N,
        ),
        dtype=np.float32,
    )

    for c in range(
        N_COSMO
    ):

        for ri, r in enumerate(
            TEST_REALIZATIONS
        ):

            full = vector_to_full(
                kappa_vec[
                    c,
                    r,
                ]
            )

            for p, (
                r0,
                r1,
            ) in enumerate(
                PATCH_ROWS
            ):

                truth[
                    c,
                    ri,
                    p,
                ] = full[
                    r0:r1,
                    :
                ]

    return truth


print()
print("Loading truth...")

TRUTH = build_truth_grid()

print(
    "TRUTH:",
    TRUTH.shape,
)


# =============================================================================
# CANONICALIZE SAVED ARRAYS
# =============================================================================

def canonicalize(
    arr,
    name,
):

    arr = np.asarray(
        arr
    )

    if arr.shape == (
        3,
        20,
        8,
        176,
        176,
    ):

        return arr.astype(
            np.float32,
            copy=False,
        )

    if arr.shape == (
        3,
        30,
        8,
        176,
        176,
    ):

        return arr[
            :,
            10:30,
            :,
            :,
            :,
        ].astype(
            np.float32,
            copy=False,
        )

    if arr.shape == (
        480,
        176,
        176,
    ):

        return arr.reshape(
            3,
            20,
            8,
            176,
            176,
        ).astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        f"{name}: unsupported shape "
        f"{arr.shape}"
    )


GS = canonicalize(
    np.load(
        GS_FILE,
        mmap_mode="r",
    ),
    "Gaussian+Starlet",
)


FUSION = canonicalize(
    np.load(
        FUSION_FILE,
        mmap_mode="r",
    ),
    "Fusion",
)


# =============================================================================
# SAFE MCALENS LOADER
# =============================================================================

BAD_KEY_WORDS = [
    "truth",
    "true",
    "mask",
    "gamma",
    "shear",
    "noise",
    "input",
    "target",
    "label",
]


SAFE_RECON_KEYS = [
    "rec",
    "recon",
    "reconstruction",
    "mcalens",
    "mcalens_rec",
    "mcalens_recon",
    "kappa_rec",
    "kappa_recon",
    "kappa_reconstruction",
    "x_rec",
    "x_recon",
    "x_final",
    "solution",
]


def inspect_first_mcalens_file():

    fn = (
        MC_ROOT
        / "C0_R10_P0.npz"
    )

    require(
        fn
    )

    print()
    print("=" * 120)
    print("MCALENS NPZ SCHEMA")
    print("=" * 120)

    print(
        fn
    )

    with np.load(
        fn
    ) as z:

        for key in z.files:

            arr = np.asarray(
                z[
                    key
                ]
            )

            print(
                f"  {key:<32} "
                f"shape={str(arr.shape):<20} "
                f"dtype={arr.dtype}"
            )


inspect_first_mcalens_file()


def choose_mcalens_key():

    global MC_KEY

    fn = (
        MC_ROOT
        / "C0_R10_P0.npz"
    )

    with np.load(
        fn
    ) as z:

        keys = list(
            z.files
        )

        if MC_KEY is not None:

            if MC_KEY not in keys:

                raise KeyError(
                    f"MC_KEY={MC_KEY!r} "
                    f"not present in {keys}"
                )

            arr = np.asarray(
                z[
                    MC_KEY
                ]
            )

            if arr.shape != (
                N,
                N,
            ):

                raise ValueError(
                    f"MC_KEY={MC_KEY!r} "
                    f"has shape {arr.shape}"
                )

            return MC_KEY


        for candidate in SAFE_RECON_KEYS:

            if candidate in keys:

                arr = np.asarray(
                    z[
                        candidate
                    ]
                )

                if arr.shape == (
                    N,
                    N,
                ):

                    return candidate


        candidates = []

        for key in keys:

            key_lower = key.lower()

            if any(
                bad in key_lower
                for bad in BAD_KEY_WORDS
            ):

                continue

            arr = np.asarray(
                z[
                    key
                ]
            )

            if arr.shape == (
                N,
                N,
            ):

                candidates.append(
                    key
                )


        if len(
            candidates
        ) == 1:

            return candidates[
                0
            ]


        raise RuntimeError(
            "\nCould not identify MCALens reconstruction safely.\n\n"
            f"Safe 176x176 candidates: {candidates}\n"
            f"All keys: {keys}\n\n"
            "Set MC_KEY explicitly near the top of the script.\n\n"
            "The code deliberately refuses to use "
            "an arbitrary 176x176 array."
        )


MC_RECON_KEY = choose_mcalens_key()


print()
print(
    "MCALens reconstruction key:",
    repr(
        MC_RECON_KEY
    )
)


def load_mcalens():

    out = np.empty(
        (
            3,
            20,
            8,
            N,
            N,
        ),
        dtype=np.float32,
    )

    for c in range(
        3
    ):

        for ri, r in enumerate(
            TEST_REALIZATIONS
        ):

            for p in range(
                8
            ):

                fn = (
                    MC_ROOT
                    / f"C{c}_R{r}_P{p}.npz"
                )

                require(
                    fn
                )

                with np.load(
                    fn
                ) as z:

                    if MC_RECON_KEY not in z:

                        raise KeyError(
                            f"{fn}: key "
                            f"{MC_RECON_KEY!r} "
                            f"is missing"
                        )

                    arr = np.asarray(
                        z[
                            MC_RECON_KEY
                        ]
                    )

                if arr.shape != (
                    N,
                    N,
                ):

                    raise ValueError(
                        f"{fn}: "
                        f"{MC_RECON_KEY} "
                        f"has shape {arr.shape}"
                    )

                out[
                    c,
                    ri,
                    p,
                ] = arr.astype(
                    np.float32
                )

    return out


print()
print("Loading MCALens...")

MC = load_mcalens()


MAPS = {
    "GAUSSIAN_STARLET": GS,
    "MCALENS": MC,
    "FUSION": FUSION,
}


# =============================================================================
# FIELD METRICS
# =============================================================================

def centered_values(
    x,
    mask,
):

    values = np.asarray(
        x,
        dtype=np.float64,
    )[
        mask
    ]

    return (
        values
        - values.mean()
    )


def field_metrics_one(
    rec,
    truth,
    mask,
):

    r = centered_values(
        rec,
        mask,
    )

    t = centered_values(
        truth,
        mask,
    )

    nmse = (
        np.sum(
            (
                r - t
            ) ** 2
        )
        /
        np.sum(
            t**2
        )
    )

    pcc = (
        np.dot(
            r,
            t,
        )
        /
        (
            np.linalg.norm(
                r
            )
            *
            np.linalg.norm(
                t
            )
            + 1e-30
        )
    )

    std_ratio = (
        np.std(
            r
        )
        /
        (
            np.std(
                t
            )
            + 1e-30
        )
    )

    return (
        float(
            nmse
        ),
        float(
            pcc
        ),
        float(
            std_ratio
        ),
    )


def field_metrics_grid(
    rec_grid,
):

    rows = []

    for c in range(
        3
    ):

        for ri, r in enumerate(
            TEST_REALIZATIONS
        ):

            for p in range(
                8
            ):

                (
                    nmse,
                    pcc,
                    std,
                ) = field_metrics_one(
                    rec_grid[
                        c,
                        ri,
                        p,
                    ],
                    TRUTH[
                        c,
                        ri,
                        p,
                    ],
                    MASK_PATCH[
                        p
                    ],
                )

                rows.append({
                    "c": c,
                    "r": r,
                    "p": p,
                    "nmse": nmse,
                    "pcc": pcc,
                    "std": std,
                })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# HARD LOADER SANITY CHECK
# =============================================================================

print()
print("=" * 120)
print("HARD MAP-LOADER SANITY CHECK")
print("=" * 120)


FIELD_CASES = {}


for method, arr in MAPS.items():

    df = field_metrics_grid(
        arr
    )

    FIELD_CASES[
        method
    ] = df

    measured = np.asarray(
        [
            df[
                "nmse"
            ].mean(),

            df[
                "pcc"
            ].mean(),

            df[
                "std"
            ].mean(),
        ]
    )

    expected = np.asarray(
        EXPECTED_MAP_METRICS[
            method
        ]
    )

    print()
    print(
        method
    )

    print(
        f"  measured NMSE = "
        f"{measured[0]:.9f}"
    )

    print(
        f"  expected NMSE = "
        f"{expected[0]:.9f}"
    )

    print(
        f"  measured PCC  = "
        f"{measured[1]:.9f}"
    )

    print(
        f"  expected PCC  = "
        f"{expected[1]:.9f}"
    )

    print(
        f"  measured std  = "
        f"{measured[2]:.9f}"
    )

    print(
        f"  expected std  = "
        f"{expected[2]:.9f}"
    )

    error = np.abs(
        measured
        - expected
    )

    tolerance = np.asarray(
        [
            5e-4,
            5e-4,
            5e-4,
        ]
    )

    if np.any(
        error
        > tolerance
    ):

        raise RuntimeError(
            f"\n{method} map loading failed "
            f"the frozen-result sanity check.\n\n"
            f"absolute differences = {error}\n\n"
            "STOPPING before producing figures.\n"
            "Most likely cause: wrong array/key "
            "loaded from reconstruction product."
        )


print()
print(
    "All frozen-map sanity checks PASSED."
)


# =============================================================================
# SAVE FIELD NUMBERS
# =============================================================================

FIELD.to_csv(
    OUT
    / "field_metrics_frozen.csv",
    index=False,
)


# =============================================================================
# FIGURE 1 — HELD-OUT EXAMPLE
# =============================================================================

def centered_full(
    x,
    mask,
):

    out = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    out -= np.mean(
        out[
            mask
        ]
    )

    return out


c = VIS_C

ri = TEST_REALIZATIONS.index(
    VIS_R
)

p = VIS_P


mask = MASK_PATCH[
    p
]


truth = centered_full(
    TRUTH[
        c,
        ri,
        p,
    ],
    mask,
)


recs = {

    method:

        centered_full(
            MAPS[
                method
            ][
                c,
                ri,
                p,
            ],
            mask,
        )

    for method in MAPS
}


map_lim = np.percentile(
    np.abs(
        truth[
            mask
        ]
    ),
    99.0,
)


res_values = np.concatenate(
    [
        (
            recs[
                method
            ]
            - truth
        )[
            mask
        ]

        for method in [
            "GAUSSIAN_STARLET",
            "MCALENS",
            "FUSION",
        ]
    ]
)


res_lim = np.percentile(
    np.abs(
        res_values
    ),
    99.0,
)


cmap = plt.get_cmap(
    "RdBu_r"
).copy()

cmap.set_bad(
    "white"
)


fig, axes = plt.subplots(
    2,
    4,
    figsize=(
        13.8,
        6.8,
    ),
    constrained_layout=True,
)


top_panels = [

    (
        "Truth",
        truth,
        None,
    ),

    (
        "Gaussian+Starlet",
        recs[
            "GAUSSIAN_STARLET"
        ],
        "GAUSSIAN_STARLET",
    ),

    (
        "MCALens",
        recs[
            "MCALENS"
        ],
        "MCALENS",
    ),

    (
        "Multiscale fusion",
        recs[
            "FUSION"
        ],
        "FUSION",
    ),
]


map_image = None


for j, (
    title,
    arr,
    method,
) in enumerate(
    top_panels
):

    map_image = axes[
        0,
        j
    ].imshow(
        np.where(
            mask,
            arr,
            np.nan,
        ),
        origin="lower",
        cmap=cmap,
        vmin=-map_lim,
        vmax=map_lim,
    )

    if method is None:

        panel_title = "Truth"

    else:

        (
            nmse,
            pcc,
            std,
        ) = field_metrics_one(
            arr,
            truth,
            mask,
        )

        panel_title = (
            f"{title}\n"
            f"NMSE={nmse:.3f}, "
            f"$r$={pcc:.3f}, "
            f"$R_\\sigma$={std:.2f}"
        )

    axes[
        0,
        j
    ].set_title(
        panel_title,
        fontsize=9,
        fontweight="bold",
    )

    axes[
        0,
        j
    ].set_axis_off()


# -------------------------------------------------------------------------
# Survey footprint mask.
#
# Keep visual convention consistent with all map panels:
#
#     WHITE = no information / unobserved
#     GRAY  = observed survey footprint
#
# mask == False -> 0 -> white
# mask == True  -> 1 -> gray
# -------------------------------------------------------------------------

mask_cmap = ListedColormap(
    [
        "white",
        "0.60",
    ]
)


axes[
    1,
    0
].imshow(
    mask.astype(
        float
    ),
    origin="lower",
    cmap=mask_cmap,
    vmin=0.0,
    vmax=1.0,
)


axes[
    1,
    0
].set_title(
    "Survey footprint mask\n"
    "white = unobserved",
    fontsize=9,
)


axes[
    1,
    0
].set_axis_off()


# -------------------------------------------------------------------------
# Residuals
# -------------------------------------------------------------------------

res_image = None


for j, method in enumerate(
    [
        "GAUSSIAN_STARLET",
        "MCALENS",
        "FUSION",
    ],
    start=1,
):

    residual = (
        recs[
            method
        ]
        - truth
    )

    res_image = axes[
        1,
        j
    ].imshow(
        np.where(
            mask,
            residual,
            np.nan,
        ),
        origin="lower",
        cmap=cmap,
        vmin=-res_lim,
        vmax=res_lim,
    )

    axes[
        1,
        j
    ].set_title(
        LABEL[
            method
        ].replace("\n+", "+").replace("\n", " ")
        + "\nresidual",
        fontsize=9,
    )

    axes[
        1,
        j
    ].set_axis_off()


cbar = fig.colorbar(
    map_image,
    ax=axes[
        0,
        :
    ],
    shrink=0.85,
    pad=0.01,
)

cbar.set_label(
    r"Centered convergence $\kappa$"
)


cbar = fig.colorbar(
    res_image,
    ax=axes[
        1,
        1:
    ],
    shrink=0.85,
    pad=0.01,
)

cbar.set_label(
    r"Residual $\hat{\kappa}-\kappa_{\rm true}$"
)


fig.savefig(
    OUT
    / "fig_maps_revised.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fig_maps_revised.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


print()
print(
    "Saved fig_maps_revised"
)


# =============================================================================
# FIGURE 2 — HELD-OUT FIELD METRICS
# =============================================================================

ORDER = [
    "KS",
    "GLIMPSE",
    "WIENER", "STARLET_NOMASK",
    "STARLET_MASK",
    "MCALENS",
    "GAUSSIAN_STARLET",
    "FUSION",
]


field_plot = (
    FIELD
    .set_index(
        "method"
    )
    .loc[
        ORDER
    ]
    .reset_index()
)


x = np.arange(
    len(
        ORDER
    )
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        15.2,
        5.1,
    ),
)


panels = [

    (
        "nmse",
        "MS-NMSE ↓",
        "Pixel-domain error",
    ),

    (
        "pcc",
        "Pearson $r$ ↑",
        "Pearson correlation",
    ),

    (
        "std",
        r"$R_\sigma=\sigma_{\rm rec}/\sigma_{\rm true}$",
        "Recovered field amplitude",
    ),
]


for ax, (
    metric,
    ylabel,
    title,
) in zip(
    axes,
    panels
):

    values = field_plot[
        metric
    ].to_numpy()

    ax.bar(
        x,
        values,
        color=[
            C[
                method
            ]
            for method in ORDER
        ],
        alpha=0.90,
        edgecolor="white",
        width=0.72,
    )

    for i, value in enumerate(
        values
    ):

        ax.text(
            i,
            value
            + (
                0.005
                if metric == "nmse"
                else 0.008
            ),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    if metric == "std":

        ax.axhline(
            1.0,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="Ideal amplitude",
        )

        ax.set_ylim(
            0.40,
            1.05,
        )

    elif metric == "nmse":

        ax.set_ylim(
            0.60,
            0.86,
        )

    elif metric == "pcc":

        ax.set_ylim(
            0.40,
            0.64,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            LABEL[
                method
            ]
            for method in ORDER
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )


fig.tight_layout()


fig.savefig(
    OUT
    / "fig_field_metrics_revised.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fig_field_metrics_revised.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    "Saved fig_field_metrics_revised"
)


# =============================================================================
# FIGURE 3 — ABLATION / INFORMATION PROGRESSION
# =============================================================================

AB_ORDER = [
    "KS",
    "STARLET_NOMASK",
    "STARLET_MASK",
    "GAUSSIAN_STARLET",
    "MCALENS",
    "FUSION",
]


ab = (
    FIELD
    .set_index(
        "method"
    )
    .loc[
        AB_ORDER
    ]
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(
        9.2,
        5.0,
    )
)


x = np.arange(
    len(
        AB_ORDER
    )
)


ax.bar(
    x,
    ab[
        "nmse"
    ],
    color=[
        C[
            method
        ]
        for method in AB_ORDER
    ],
    width=0.70,
    edgecolor="white",
)


for i, (
    method,
    value,
) in enumerate(
    zip(
        AB_ORDER,
        ab[
            "nmse"
        ],
    )
):

    ax.text(
        i,
        value + 0.003,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        "KS",
        "Starlet\nno mask",
        "Starlet\nmask",
        "Gaussian\n+Starlet",
        "MCALens\ncomparator",
        "Multiscale\nfusion",
    ]
)


ax.set_ylabel(
    "Held-out MS-NMSE ↓"
)


ax.set_ylim(
    0.60,
    0.75,
)


ax.grid(
    axis="y",
    alpha=0.25,
)


ax.axvline(
    3.5,
    color="0.75",
    linestyle="--",
    linewidth=1.0,
)


ax.text(
    1.5,
    0.747,
    "Sparse / Gaussian+Sparse progression",
    ha="center",
    va="top",
    fontsize=9,
)


ax.text(
    4.5,
    0.747,
    "Comparator + fusion",
    ha="center",
    va="top",
    fontsize=9,
)


fig.tight_layout()


fig.savefig(
    OUT
    / "fig_ablation_revised.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fig_ablation_revised.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    "Saved fig_ablation_revised"
)


# =============================================================================
# SPECTRAL FIDELITY
# =============================================================================

theta_pix_rad = np.deg2rad(
    PIX_ARCMIN
    / 60.0
)


frequency = np.fft.fftfreq(
    N
)


FY, FX = np.meshgrid(
    frequency,
    frequency,
    indexing="ij",
)


ELL_MAP = (
    2.0
    * np.pi
    * np.sqrt(
        FX**2
        + FY**2
    )
    / theta_pix_rad
)


nonzero_ell = ELL_MAP[
    ELL_MAP > 0
]


ELL_MIN = max(
    60.0,
    np.percentile(
        nonzero_ell,
        0.1,
    ),
)


ELL_MAX = min(
    6000.0,
    np.percentile(
        nonzero_ell,
        99.0,
    ),
)


ELL_EDGES = np.geomspace(
    ELL_MIN,
    ELL_MAX,
    N_ELL_BINS
    + 1,
)


ELL = np.sqrt(
    ELL_EDGES[
        :-1
    ]
    * ELL_EDGES[
        1:
    ]
)


def centered_zero_fill(
    x,
    mask,
):

    output = np.zeros(
        x.shape,
        dtype=np.float64,
    )

    mean_value = np.mean(
        x[
            mask
        ]
    )

    output[
        mask
    ] = (
        x[
            mask
        ]
        - mean_value
    )

    return output


def radial_average(
    field2d,
):

    result = np.full(
        N_ELL_BINS,
        np.nan,
        dtype=np.float64,
    )

    for ib in range(
        N_ELL_BINS
    ):

        selection = (
            (
                ELL_MAP
                >= ELL_EDGES[
                    ib
                ]
            )
            &
            (
                ELL_MAP
                < ELL_EDGES[
                    ib + 1
                ]
            )
        )

        if np.any(
            selection
        ):

            result[
                ib
            ] = np.mean(
                field2d[
                    selection
                ]
            )

    return result


def spectral_blocks(
    rec_grid,
):

    all_T = []
    all_A = []
    all_R = []

    for ri in range(
        N_TEST
    ):

        sum_tt = np.zeros(
            N_ELL_BINS
        )

        sum_rr = np.zeros(
            N_ELL_BINS
        )

        sum_rt = np.zeros(
            N_ELL_BINS
        )

        sum_w = np.zeros(
            N_ELL_BINS
        )

        for c in range(
            3
        ):

            for p in range(
                8
            ):

                mask = MASK_PATCH[
                    p
                ]

                r = centered_zero_fill(
                    rec_grid[
                        c,
                        ri,
                        p,
                    ],
                    mask,
                )

                t = centered_zero_fill(
                    TRUTH[
                        c,
                        ri,
                        p,
                    ],
                    mask,
                )

                R = np.fft.fft2(
                    r,
                    norm="ortho",
                )

                T = np.fft.fft2(
                    t,
                    norm="ortho",
                )

                Ctt = radial_average(
                    np.real(
                        T
                        * np.conj(
                            T
                        )
                    )
                )

                Crr = radial_average(
                    np.real(
                        R
                        * np.conj(
                            R
                        )
                    )
                )

                Crt = radial_average(
                    np.real(
                        R
                        * np.conj(
                            T
                        )
                    )
                )

                weight = float(
                    mask.sum()
                )

                valid = (
                    np.isfinite(
                        Ctt
                    )
                    &
                    np.isfinite(
                        Crr
                    )
                    &
                    np.isfinite(
                        Crt
                    )
                )

                sum_tt[
                    valid
                ] += (
                    weight
                    * Ctt[
                        valid
                    ]
                )

                sum_rr[
                    valid
                ] += (
                    weight
                    * Crr[
                        valid
                    ]
                )

                sum_rt[
                    valid
                ] += (
                    weight
                    * Crt[
                        valid
                    ]
                )

                sum_w[
                    valid
                ] += weight


        Ctt = (
            sum_tt
            / np.maximum(
                sum_w,
                1e-30,
            )
        )

        Crr = (
            sum_rr
            / np.maximum(
                sum_w,
                1e-30,
            )
        )

        Crt = (
            sum_rt
            / np.maximum(
                sum_w,
                1e-30,
            )
        )


        transfer = (
            Crt
            / (
                Ctt
                + 1e-30
            )
        )


        amplitude = (
            Crr
            / (
                Ctt
                + 1e-30
            )
        )


        correlation = (
            Crt
            /
            (
                np.sqrt(
                    Crr
                    * Ctt
                )
                + 1e-30
            )
        )


        all_T.append(
            transfer
        )

        all_A.append(
            amplitude
        )

        all_R.append(
            correlation
        )


    return (
        np.asarray(
            all_T
        ),
        np.asarray(
            all_A
        ),
        np.asarray(
            all_R
        ),
    )


print()
print("=" * 120)
print("SPECTRAL DIAGNOSTICS")
print("=" * 120)


SPECTRAL = {}

# =============================================================================
# V9 ONLY — Wiener for spectral diagnostics
#
# Keep the global MAPS dictionary unchanged so Wiener is NOT propagated into
# peak diagnostics.
# =============================================================================

V9_SPECTRAL_MAPS = dict(MAPS)
V9_SPECTRAL_MAPS["WIENER"] = V8_WIENER_MAPS


for method in [
    "WIENER", "GAUSSIAN_STARLET",
    "MCALENS",
    "FUSION",
]:

    print(
        "Computing",
        method,
        flush=True,
    )

    SPECTRAL[
        method
    ] = spectral_blocks(
        V9_SPECTRAL_MAPS[method]
    )

    (
        Tcurve,
        Acurve,
        Rcurve,
    ) = SPECTRAL[
        method
    ]


    mean_T_error = np.nanmean(
        np.abs(
            Tcurve
            - 1.0
        )
    )


    mean_A_error = np.nanmean(
        np.abs(
            Acurve
            - 1.0
        )
    )


    mean_r = np.nanmean(
        Rcurve
    )


    print(
        f"  mean |1-T| = "
        f"{mean_T_error:.5f}"
    )

    print(
        f"  mean |1-A| = "
        f"{mean_A_error:.5f}"
    )

    print(
        f"  mean r_l   = "
        f"{mean_r:.5f}"
    )


    if method == "MCALENS":

        if (
            mean_T_error
            < 0.01
            and
            mean_A_error
            < 0.01
        ):

            raise RuntimeError(
                "\nMCALens appears spectrally identical "
                "to truth.\n"
                "This strongly suggests an incorrect "
                "reconstruction array was loaded."
            )


# =============================================================================


# V9 spectral plotting metadata
C.setdefault(
    "WIENER",
    "#CC79A7",
)

LABEL.setdefault(
    "WIENER",
    "Wiener",
)

# FIGURE 4 — SPECTRAL FIDELITY
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        14.2,
        4.5,
    ),
)


panels = [
    (
        0,
        r"$T_\ell$",
        "Cross-transfer",
    ),

    (
        1,
        r"$A_\ell$",
        "Auto-power ratio",
    ),

    (
        2,
        r"$r_\ell$",
        "Spectral correlation",
    ),
]


for ax, (
    index,
    ylabel,
    title,
) in zip(
    axes,
    panels
):

    for method in [
        "WIENER", "GAUSSIAN_STARLET",
        "MCALENS",
        "FUSION",
    ]:

        values = SPECTRAL[
            method
        ][
            index
        ]

        median = np.nanmedian(
            values,
            axis=0,
        )

        p16 = np.nanpercentile(
            values,
            16,
            axis=0,
        )

        p84 = np.nanpercentile(
            values,
            84,
            axis=0,
        )

        ax.plot(
            ELL,
            median,
            color=C[
                method
            ],
            linewidth=2.0,
            label=LABEL[
                method
            ].replace("\n+", "+").replace("\n", " "),
        )

        ax.fill_between(
            ELL,
            p16,
            p84,
            color=C[
                method
            ],
            alpha=0.10,
        )


    ax.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.1,
        label=(
            "Ideal"
            if index == 0
            else None
        ),
    )


    ax.set_xscale(
        "log"
    )


    ax.set_xlabel(
        r"Multipole $\ell$"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title,
        fontweight="bold",
    )


    ax.grid(
        alpha=0.25,
        which="both",
    )


axes[
    0
].set_ylim(
    SPECTRAL_AMPLITUDE_YMIN,
    SPECTRAL_AMPLITUDE_YMAX,
)


axes[
    1
].set_ylim(
    SPECTRAL_AMPLITUDE_YMIN,
    SPECTRAL_AMPLITUDE_YMAX,
)


axes[
    2
].set_ylim(
    SPECTRAL_CORR_YMIN,
    SPECTRAL_CORR_YMAX,
)


axes[
    2
].legend(
    loc="lower left",
    fontsize=8,
)


fig.tight_layout()


fig.savefig(
    OUT
    / "fig_spectral_revised.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fig_spectral_revised.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    "Saved fig_spectral_revised"
)


# =============================================================================
# PEAK HELPERS
# =============================================================================

def mask_normalized_smooth(
    image,
    mask,
    sigma_pix,
):

    image = np.asarray(
        image,
        dtype=np.float64,
    )

    mask_float = mask.astype(
        np.float64
    )

    numerator = gaussian_filter(
        image
        * mask_float,
        sigma=sigma_pix,
        mode="constant",
        cval=0.0,
    )

    denominator = gaussian_filter(
        mask_float,
        sigma=sigma_pix,
        mode="constant",
        cval=0.0,
    )

    valid = (
        denominator
        >= MASK_SUPPORT_MIN
    )

    output = np.zeros_like(
        numerator
    )

    output[
        valid
    ] = (
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    return (
        output,
        valid,
    )


def find_peaks(
    field,
    valid,
):

    local_max = maximum_filter(
        field,
        size=3,
        mode="constant",
        cval=-np.inf,
    )

    is_peak = (
        valid
        &
        np.isfinite(
            field
        )
        &
        (
            field
            == local_max
        )
    )

    coordinates = np.argwhere(
        is_peak
    )

    values = field[
        is_peak
    ]

    return (
        coordinates,
        values,
    )


def exact_match_count(
    truth_coords,
    rec_coords,
    radius_pix,
):

    n_truth = len(
        truth_coords
    )

    n_rec = len(
        rec_coords
    )

    if (
        n_truth == 0
        or
        n_rec == 0
    ):

        return 0


    dy = (
        truth_coords[
            :,
            None,
            0
        ]
        -
        rec_coords[
            None,
            :,
            0
        ]
    )


    dx = (
        truth_coords[
            :,
            None,
            1
        ]
        -
        rec_coords[
            None,
            :,
            1
        ]
    )


    adjacency = (
        (
            dx**2
            + dy**2
        )
        <= radius_pix**2
    )


    if not np.any(
        adjacency
    ):

        return 0


    graph = csr_matrix(
        adjacency.astype(
            np.int8
        )
    )


    matching = maximum_bipartite_matching(
        graph,
        perm_type="column",
    )


    return int(
        np.sum(
            matching
            >= 0
        )
    )


def nearest_amplitude_pairs(
    truth_coords,
    rec_coords,
    radius_pix,
):

    if (
        len(
            truth_coords
        ) == 0
        or
        len(
            rec_coords
        ) == 0
    ):

        return []


    dy = (
        truth_coords[
            :,
            None,
            0
        ]
        -
        rec_coords[
            None,
            :,
            0
        ]
    )


    dx = (
        truth_coords[
            :,
            None,
            1
        ]
        -
        rec_coords[
            None,
            :,
            1
        ]
    )


    d2 = (
        dx**2
        + dy**2
    )


    candidate_pairs = np.argwhere(
        d2
        <= radius_pix**2
    )


    if len(
        candidate_pairs
    ) == 0:

        return []


    distance_values = d2[
        candidate_pairs[
            :,
            0
        ],
        candidate_pairs[
            :,
            1
        ],
    ]


    order = np.argsort(
        distance_values
    )


    used_truth = set()
    used_rec = set()

    pairs = []


    for kk in order:

        ti = int(
            candidate_pairs[
                kk,
                0
            ]
        )


        ri = int(
            candidate_pairs[
                kk,
                1
            ]
        )


        if (
            ti in used_truth
            or
            ri in used_rec
        ):

            continue


        used_truth.add(
            ti
        )

        used_rec.add(
            ri
        )


        pairs.append(
            (
                ti,
                ri,
            )
        )


    return pairs


# =============================================================================
# PEAK ANALYSIS
# =============================================================================

print()
print("=" * 120)
print("PEAK COMPLETENESS / PURITY")
print("=" * 120)


PEAK = {}

PEAK_AMP = {}


for scale_arcmin in SMOOTH_ARCMIN:

    sigma_pix = (
        scale_arcmin
        / PIX_ARCMIN
    )


    total_truth = 0


    predicted = {

        method:

            np.zeros(
                len(
                    REC_NU_THRESHOLDS
                ),
                dtype=np.int64,
            )

        for method in MAPS
    }


    matched = {

        method:

            np.zeros(
                len(
                    REC_NU_THRESHOLDS
                ),
                dtype=np.int64,
            )

        for method in MAPS
    }


    amplitude_ratios = {

        method:
            []

        for method in MAPS
    }


    for c in range(
        3
    ):

        for ri in range(
            20
        ):

            for p in range(
                8
            ):

                mask = MASK_PATCH[
                    p
                ]


                (
                    truth_smoothed,
                    truth_valid,
                ) = mask_normalized_smooth(
                    TRUTH[
                        c,
                        ri,
                        p,
                    ],
                    mask,
                    sigma_pix,
                )


                sigma_truth = (
                    np.std(
                        truth_smoothed[
                            truth_valid
                        ]
                    )
                    + 1e-30
                )


                truth_nu = (
                    truth_smoothed
                    / sigma_truth
                )


                (
                    truth_coords_all,
                    truth_values_all,
                ) = find_peaks(
                    truth_nu,
                    truth_valid,
                )


                keep_truth_pr = (
                    truth_values_all
                    >= TRUTH_NU_PR
                )


                truth_coords_pr = (
                    truth_coords_all[
                        keep_truth_pr
                    ]
                )


                total_truth += len(
                    truth_coords_pr
                )


                keep_truth_amp = (
                    truth_values_all
                    >= TRUTH_NU_AMP
                )


                truth_coords_amp = (
                    truth_coords_all[
                        keep_truth_amp
                    ]
                )


                for method, grid in MAPS.items():

                    (
                        rec_smoothed,
                        rec_valid,
                    ) = mask_normalized_smooth(
                        grid[
                            c,
                            ri,
                            p,
                        ],
                        mask,
                        sigma_pix,
                    )


                    valid = (
                        truth_valid
                        &
                        rec_valid
                    )


                    rec_nu = (
                        rec_smoothed
                        / sigma_truth
                    )


                    (
                        rec_coords_all,
                        rec_values_all,
                    ) = find_peaks(
                        rec_nu,
                        valid,
                    )


                    for threshold_index, threshold in enumerate(
                        REC_NU_THRESHOLDS
                    ):

                        keep_rec = (
                            rec_values_all
                            >= threshold
                        )


                        rec_coords = (
                            rec_coords_all[
                                keep_rec
                            ]
                        )


                        predicted[
                            method
                        ][
                            threshold_index
                        ] += len(
                            rec_coords
                        )


                        matched[
                            method
                        ][
                            threshold_index
                        ] += exact_match_count(
                            truth_coords_pr,
                            rec_coords,
                            MATCH_RADIUS_PIX,
                        )


                    pairs = nearest_amplitude_pairs(
                        truth_coords_amp,
                        rec_coords_all,
                        MATCH_RADIUS_PIX,
                    )


                    for truth_index, rec_index in pairs:

                        yt, xt = (
                            truth_coords_amp[
                                truth_index
                            ]
                        )


                        yr, xr = (
                            rec_coords_all[
                                rec_index
                            ]
                        )


                        truth_amp = (
                            truth_smoothed[
                                yt,
                                xt
                            ]
                        )


                        rec_amp = (
                            rec_smoothed[
                                yr,
                                xr
                            ]
                        )


                        if abs(
                            truth_amp
                        ) > 1e-30:

                            amplitude_ratios[
                                method
                            ].append(
                                rec_amp
                                / truth_amp
                            )


    # =============================================================================
    # BUILD COMPLETENESS-PURITY CURVES
    # =============================================================================

    curves = {}


    for method in MAPS:

        completeness = (
            matched[
                method
            ]
            /
            max(
                total_truth,
                1
            )
        )


        purity = np.divide(
            matched[
                method
            ],
            predicted[
                method
            ],
            out=np.full(
                len(
                    REC_NU_THRESHOLDS
                ),
                np.nan,
            ),
            where=(
                predicted[
                    method
                ]
                > 0
            ),
        )


        curves[
            method
        ] = (
            completeness,
            purity,
        )


    # =============================================================================
    # COMMON COMPLETENESS RANGE
    # =============================================================================

    valid_ranges = []


    for method, (
        completeness,
        purity,
    ) in curves.items():

        valid = (
            np.isfinite(
                completeness
            )
            &
            np.isfinite(
                purity
            )
        )


        if np.sum(
            valid
        ) < 2:

            continue


        valid_ranges.append(
            (
                np.min(
                    completeness[
                        valid
                    ]
                ),

                np.max(
                    completeness[
                        valid
                    ]
                ),
            )
        )


    common_lo = max(
        value[
            0
        ]
        for value in valid_ranges
    )


    common_hi = min(
        value[
            1
        ]
        for value in valid_ranges
    )


    if common_hi <= common_lo:

        raise RuntimeError(
            f"No common completeness interval "
            f"at {scale_arcmin} arcmin"
        )


    # =============================================================================
    # COMMON-RANGE NORMALIZED AUC
    # =============================================================================

    AUC = {}


    for method, (
        completeness,
        purity,
    ) in curves.items():

        valid = (
            np.isfinite(
                completeness
            )
            &
            np.isfinite(
                purity
            )
            &
            (
                completeness
                >= common_lo
            )
            &
            (
                completeness
                <= common_hi
            )
        )


        cc = completeness[
            valid
        ]

        pp = purity[
            valid
        ]


        order = np.argsort(
            cc
        )

        cc = cc[
            order
        ]

        pp = pp[
            order
        ]


        unique_c = np.unique(
            cc
        )


        collapsed_purity = []


        for value in unique_c:

            indices = np.where(
                cc
                == value
            )[
                0
            ]


            collapsed_purity.append(
                pp[
                    indices[
                        -1
                    ]
                ]
            )


        unique_p = np.asarray(
            collapsed_purity
        )


        if len(
            unique_c
        ) < 2:

            auc = np.nan


        else:

            auc = (
                np.trapz(
                    unique_p,
                    unique_c,
                )
                /
                (
                    common_hi
                    - common_lo
                )
            )


        AUC[
            method
        ] = float(
            auc
        )


    # =============================================================================
    # PEAK AMPLITUDE
    # =============================================================================

    PEAK_AMP[
        scale_arcmin
    ] = {

        method:

            float(
                np.median(
                    amplitude_ratios[
                        method
                    ]
                )
            )

        if len(
            amplitude_ratios[
                method
            ]
        ) > 0

        else np.nan

        for method in MAPS
    }


    PEAK[
        scale_arcmin
    ] = {
        "curves":
            curves,

        "auc":
            AUC,

        "truth_total":
            total_truth,

        "common_lo":
            common_lo,

        "common_hi":
            common_hi,
    }


    print()
    print(
        f"{scale_arcmin:.0f} arcmin"
    )

    print(
        "  truth peaks:",
        total_truth
    )


    for method in [
        "GAUSSIAN_STARLET",
        "MCALENS",
        "FUSION",
    ]:

        completeness, purity = curves[
            method
        ]


        print(
            f"  {method:<18} "
            f"Cmax="
            f"{np.nanmax(completeness):.4f}  "
            f"Pmin="
            f"{np.nanmin(purity):.4f}  "
            f"Pmax="
            f"{np.nanmax(purity):.4f}  "
            f"AUC="
            f"{AUC[method]:.5f}  "
            f"amp="
            f"{PEAK_AMP[scale_arcmin][method]:.5f}"
        )


    mc_completeness, mc_purity = curves[
        "MCALENS"
    ]


    valid_mc = np.isfinite(
        mc_purity
    )


    if (
        np.sum(
            valid_mc
        ) > 10
        and
        np.nanmax(
            np.abs(
                mc_purity[
                    valid_mc
                ]
                - 1.0
            )
        ) < 1e-8
    ):

        raise RuntimeError(
            "\nMCALens purity is exactly 1 "
            "across the entire threshold curve.\n"
            "STOPPING: this is treated as "
            "a diagnostic failure, not a result."
        )


# =============================================================================
# FIGURE 5 — PEAK COMPLETENESS / PURITY
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        13.8,
        4.4,
    ),
    sharex=True,
    sharey=True,
)


for ax, scale in zip(
    axes,
    SMOOTH_ARCMIN
):

    data = PEAK[
        scale
    ]


    for method in [
        "GAUSSIAN_STARLET",
        "MCALENS",
        "FUSION",
    ]:

        (
            completeness,
            purity,
        ) = data[
            "curves"
        ][
            method
        ]


        valid = (
            np.isfinite(
                completeness
            )
            &
            np.isfinite(
                purity
            )
        )


        ax.plot(
            completeness[
                valid
            ],
            purity[
                valid
            ],
            color=C[
                method
            ],
            linewidth=2.0,
            label=(
                LABEL[
                    method
                ].replace("\n+", "+").replace("\n", " ")
                +
                f"  AUC="
                f"{data['auc'][method]:.3f}"
            ),
        )


    ax.axvspan(
        data[
            "common_lo"
        ],
        data[
            "common_hi"
        ],
        color="0.5",
        alpha=0.05,
    )


    ax.set_xlabel(
        "Completeness"
    )


    ax.set_title(
        f"{scale:.0f} arcmin",
        fontweight="bold",
    )


    ax.grid(
        alpha=0.25,
    )


axes[
    0
].set_ylabel(
    "Purity"
)


axes[
    2
].legend(
    loc="lower left",
    fontsize=8,
)


fig.tight_layout()


fig.savefig(
    OUT
    / "fig_peak_pr_revised.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fig_peak_pr_revised.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    "Saved fig_peak_pr_revised"
)


# =============================================================================
# FIGURE 6 — PEAK AMPLITUDE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(
        7.7,
        4.9,
    )
)


x = np.arange(
    len(
        SMOOTH_ARCMIN
    )
)


width = 0.23


methods = [
    "GAUSSIAN_STARLET",
    "MCALENS",
    "FUSION",
]


for method_index, method in enumerate(
    methods
):

    values = [
        PEAK_AMP[
            scale
        ][
            method
        ]
        for scale in SMOOTH_ARCMIN
    ]


    xpos = (
        x
        +
        (
            method_index
            - 1
        )
        * width
    )


    ax.bar(
        xpos,
        values,
        width=width,
        color=C[
            method
        ],
        label=LABEL[
            method
        ].replace("\n+", "+").replace("\n", " "),
        edgecolor="white",
    )


    for xx, value in zip(
        xpos,
        values,
    ):

        ax.text(
            xx,
            value + 0.012,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )


ax.axhline(
    1.0,
    color="black",
    linestyle=":",
    linewidth=1.2,
    label="Ideal amplitude",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        "2′",
        "4′",
        "6′",
    ]
)


ax.set_ylabel(
    "Median recovered / true "
    "peak amplitude"
)


ax.set_xlabel(
    "Gaussian smoothing scale"
)


ax.set_ylim(
    0.40,
    1.08,
)


ax.grid(
    axis="y",
    alpha=0.25,
)


ax.legend()


fig.tight_layout()


fig.savefig(
    OUT
    / "fig_peak_amplitude_revised.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT
    / "fig_peak_amplitude_revised.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close(
    fig
)


print(
    "Saved fig_peak_amplitude_revised"
)


# =============================================================================
# SAVE SPECTRAL DATA
# =============================================================================

spectral_rows = []


for method in SPECTRAL:

    (
        transfer,
        amplitude,
        correlation,
    ) = SPECTRAL[
        method
    ]


    for block in range(
        N_TEST
    ):

        for ib in range(
            N_ELL_BINS
        ):

            spectral_rows.append({

                "method":
                    method,

                "block":
                    block,

                "ell":
                    ELL[
                        ib
                    ],

                "T":
                    transfer[
                        block,
                        ib
                    ],

                "A":
                    amplitude[
                        block,
                        ib
                    ],

                "r":
                    correlation[
                        block,
                        ib
                    ],
            })


pd.DataFrame(
    spectral_rows
).to_csv(
    OUT
    / "spectral_curves.csv",
    index=False,
)


# =============================================================================
# SAVE PEAK DATA
# =============================================================================

peak_rows = []


for scale in SMOOTH_ARCMIN:

    for method in MAPS:

        (
            completeness,
            purity,
        ) = PEAK[
            scale
        ][
            "curves"
        ][
            method
        ]


        for threshold, cc, pp in zip(
            REC_NU_THRESHOLDS,
            completeness,
            purity,
        ):

            peak_rows.append({

                "scale_arcmin":
                    scale,

                "method":
                    method,

                "threshold_rec":
                    threshold,

                "completeness":
                    cc,

                "purity":
                    pp,

                "auc_common":
                    PEAK[
                        scale
                    ][
                        "auc"
                    ][
                        method
                    ],

                "common_completeness_min":
                    PEAK[
                        scale
                    ][
                        "common_lo"
                    ],

                "common_completeness_max":
                    PEAK[
                        scale
                    ][
                        "common_hi"
                    ],
            })


pd.DataFrame(
    peak_rows
).to_csv(
    OUT
    / "peak_completeness_purity.csv",
    index=False,
)


amp_rows = []


for scale in SMOOTH_ARCMIN:

    for method in MAPS:

        amp_rows.append({

            "scale_arcmin":
                scale,

            "method":
                method,

            "peak_amplitude_ratio":
                PEAK_AMP[
                    scale
                ][
                    method
                ],
        })


pd.DataFrame(
    amp_rows
).to_csv(
    OUT
    / "peak_amplitude.csv",
    index=False,
)


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 120)
print("FIGURES WRITTEN")
print("=" * 120)


for filename in [
    "fig_maps_revised.pdf",
    "fig_field_metrics_revised.pdf",
    "fig_ablation_revised.pdf",
    "fig_spectral_revised.pdf",
    "fig_peak_pr_revised.pdf",
    "fig_peak_amplitude_revised.pdf",
]:

    print(
        OUT
        / filename
    )


print()
print("NUMERICAL OUTPUT")


for filename in [
    "field_metrics_frozen.csv",
    "spectral_curves.csv",
    "peak_completeness_purity.csv",
    "peak_amplitude.csv",
]:

    print(
        OUT
        / filename
    )


print()
print("DONE")
