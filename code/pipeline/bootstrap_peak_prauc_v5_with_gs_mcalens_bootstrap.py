#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bootstrap realization-block confidence intervals for the V5
peak completeness-purity statistic.

Block:
    one realization index
    containing 3 cosmologies x 8 patches.

N independent blocks:
    20

Comparisons:
    Fusion - Gaussian+Starlet
    Fusion - MCALens
    Gaussian+Starlet - MCALens

IMPORTANT
---------
MCALens reconstruction is read explicitly from key:

    "mcalens"

No generic NPZ-key fallback is used.
"""
from project_paths import PATHS

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import (
    gaussian_filter,
    maximum_filter,
)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


# =============================================================================
# CONFIG
# =============================================================================

N = 176
PIX_ARCMIN = 2.0

TEST_R = list(
    range(
        10,
        30,
    )
)

N_BLOCKS = 20

MASK_SUPPORT_MIN = 0.70

TRUTH_NU = 2.5

MATCH_RADIUS_ARCMIN = 4.0

MATCH_RADIUS_PIX = (
    MATCH_RADIUS_ARCMIN
    / PIX_ARCMIN
)

SMOOTH = [
    2.0,
    4.0,
    6.0,
]

THRESHOLDS = np.linspace(
    8.0,
    -0.5,
    120,
)

N_BOOT = 10000

SEED = 123456

MCALENS_KEY = "mcalens"


# =============================================================================
# PATHS
# =============================================================================

DATA = Path(
    str(PATHS.data_root)
)

MASK_FILE = (
    DATA
    / "WIDE12H_bin2_2arcmin_mask.npy"
)

KAPPA_FILE = (
    DATA
    / "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
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
    str(PATHS.work_root / 'test/peak_bootstrap_v5')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# INPUT CHECKS
# =============================================================================

for fn in [
    MASK_FILE,
    KAPPA_FILE,
    GS_FILE,
    FUSION_FILE,
]:

    if not fn.exists():

        raise FileNotFoundError(
            fn
        )


if not MC_ROOT.exists():

    raise FileNotFoundError(
        MC_ROOT
    )


# =============================================================================
# LOAD TRUTH / MASK
# =============================================================================

mask_full = np.load(
    MASK_FILE
).astype(
    bool
)

kappa_vec = np.load(
    KAPPA_FILE,
    mmap_mode="r",
)


PATCH_ROWS = [

    (
        p * N,
        (p + 1) * N,
    )

    for p in range(
        8
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


def vector_to_full(
    vec,
):

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


TRUTH = np.empty(

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
        TEST_R
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

            TRUTH[
                c,
                ri,
                p,
            ] = full[
                r0:r1,
                :
            ]


# =============================================================================
# CANONICALIZE ARRAYS
# =============================================================================

def canonicalize(
    arr,
):

    arr = np.asarray(
        arr
    )


    if arr.shape == (

        3,
        20,
        8,
        N,
        N,

    ):

        return arr.astype(
            np.float32,
            copy=False,
        )


    if arr.shape == (

        3,
        30,
        8,
        N,
        N,

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
        N,
        N,

    ):

        return arr.reshape(

            3,
            20,
            8,
            N,
            N,

        ).astype(
            np.float32,
            copy=False,
        )


    raise ValueError(
        f"Unsupported array shape: {arr.shape}"
    )


GS = canonicalize(

    np.load(
        GS_FILE,
        mmap_mode="r",
    )

)


FUSION = canonicalize(

    np.load(
        FUSION_FILE,
        mmap_mode="r",
    )

)


# =============================================================================
# LOAD MCALENS — EXPLICIT KEY
# =============================================================================

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


    first_schema_printed = False


    for c in range(
        3
    ):

        for ri, r in enumerate(
            TEST_R
        ):

            for p in range(
                8
            ):

                fn = (

                    MC_ROOT
                    / f"C{c}_R{r}_P{p}.npz"

                )


                if not fn.exists():

                    raise FileNotFoundError(
                        fn
                    )


                with np.load(
                    fn
                ) as z:

                    if not first_schema_printed:

                        print()
                        print(
                            "=" * 120
                        )

                        print(
                            "MCALENS NPZ SCHEMA"
                        )

                        print(
                            "=" * 120
                        )

                        print(
                            fn
                        )

                        for key in z.files:

                            a = np.asarray(
                                z[
                                    key
                                ]
                            )

                            print(
                                f"  {key:<20} "
                                f"shape={a.shape} "
                                f"dtype={a.dtype}"
                            )

                        first_schema_printed = True


                    if MCALENS_KEY not in z:

                        raise KeyError(

                            f"{fn}\n"
                            f"Expected MCALens reconstruction key "
                            f"{MCALENS_KEY!r}.\n"
                            f"Available keys: {z.files}"

                        )


                    a = np.asarray(

                        z[
                            MCALENS_KEY
                        ]

                    )


                if a.shape != (
                    N,
                    N,
                ):

                    raise ValueError(

                        f"{fn}: key {MCALENS_KEY!r} "
                        f"has shape {a.shape}, "
                        f"expected {(N, N)}"

                    )


                out[
                    c,
                    ri,
                    p,
                ] = a.astype(
                    np.float32
                )


    return out


MC = load_mcalens()


MAPS = {

    "GAUSSIAN_STARLET":
        GS,

    "MCALENS":
        MC,

    "FUSION":
        FUSION,
}


# =============================================================================
# FIELD SANITY CHECK
# =============================================================================

def field_metrics(
    rec,
    truth,
    mask,
):

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[
        mask
    ]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[
        mask
    ]


    r -= r.mean()
    t -= t.mean()


    nmse = (

        np.sum(
            (r - t) ** 2
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


    std = (

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
        nmse,
        pcc,
        std,
    )


def mean_field_metrics(
    grid,
):

    vals = []


    for c in range(
        3
    ):

        for ri in range(
            20
        ):

            for p in range(
                8
            ):

                vals.append(

                    field_metrics(

                        grid[
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
                )


    return np.mean(
        np.asarray(
            vals
        ),
        axis=0,
    )


EXPECTED = {

    "GAUSSIAN_STARLET":
        np.array(
            [
                0.642805480,
                0.59660041,
                0.58345016,
            ]
        ),

    "MCALENS":
        np.array(
            [
                0.651369139,
                0.595698,
                0.685784,
            ]
        ),

    "FUSION":
        np.array(
            [
                0.634196,
                0.603865,
                0.631480,
            ]
        ),
}


print()
print(
    "=" * 120
)

print(
    "FROZEN MAP SANITY CHECK"
)

print(
    "=" * 120
)


for method, grid in MAPS.items():

    measured = mean_field_metrics(
        grid
    )

    expected = EXPECTED[
        method
    ]

    print()
    print(
        method
    )

    print(
        "  measured:",
        measured
    )

    print(
        "  expected:",
        expected
    )

    print(
        "  abs diff:",
        np.abs(
            measured
            - expected
        )
    )


    if np.any(

        np.abs(
            measured
            - expected
        )

        > 5e-4

    ):

        raise RuntimeError(

            f"{method} failed frozen-map sanity check."

        )


print()
print(
    "All map sanity checks PASSED."
)


# =============================================================================
# PEAK HELPERS
# =============================================================================

def smooth_masked(
    x,
    mask,
    sigma,
):

    mf = mask.astype(
        float
    )


    num = gaussian_filter(

        x
        * mf,

        sigma=sigma,

        mode="constant",

        cval=0.0,
    )


    den = gaussian_filter(

        mf,

        sigma=sigma,

        mode="constant",

        cval=0.0,
    )


    valid = (

        den
        >= MASK_SUPPORT_MIN

    )


    out = np.zeros_like(
        num
    )


    out[
        valid
    ] = (

        num[
            valid
        ]

        /

        den[
            valid
        ]
    )


    return (
        out,
        valid,
    )


def peaks(
    field,
    valid,
):

    mx = maximum_filter(

        field,

        size=3,

        mode="constant",

        cval=-np.inf,
    )


    selection = (

        valid

        &

        np.isfinite(
            field
        )

        &

        (
            field
            == mx
        )
    )


    return (

        np.argwhere(
            selection
        ),

        field[
            selection
        ],
    )


def match_count(
    truth_coords,
    rec_coords,
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

        <= MATCH_RADIUS_PIX**2
    )


    if not adjacency.any():

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


def curve_from_counts(
    truth_total,
    predicted,
    matched,
):

    completeness = (

        matched

        /

        max(
            truth_total,
            1,
        )
    )


    purity = np.divide(

        matched,
        predicted,

        out=np.full(
            len(
                predicted
            ),
            np.nan,
            dtype=float,
        ),

        where=(
            predicted > 0
        ),
    )


    return (
        completeness,
        purity,
    )


def normalized_common_auc(
    curves,
):

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

            return {
                m: np.nan
                for m in curves
            }


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

        x[
            0
        ]

        for x in valid_ranges
    )


    common_hi = min(

        x[
            1
        ]

        for x in valid_ranges
    )


    if common_hi <= common_lo:

        return {

            method:
                np.nan

            for method in curves
        }


    output = {}


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


        unique_p = []


        for value in unique_c:

            indices = np.where(

                cc
                == value

            )[
                0
            ]


            # Same V5 convention:
            # keep the final threshold point,
            # NOT the optimistic maximum purity.

            unique_p.append(

                pp[
                    indices[
                        -1
                    ]
                ]

            )


        unique_p = np.asarray(
            unique_p
        )


        if len(
            unique_c
        ) < 2:

            output[
                method
            ] = np.nan


        else:

            output[
                method
            ] = (

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


    return output


# =============================================================================
# PRECOMPUTE REALIZATION-BLOCK COUNTS
# =============================================================================

BLOCK = {}


for scale in SMOOTH:

    sigma = (

        scale
        / PIX_ARCMIN

    )


    BLOCK[
        scale
    ] = {}


    for method in MAPS:

        BLOCK[
            scale
        ][
            method
        ] = {

            "truth":

                np.zeros(
                    N_BLOCKS,
                    dtype=np.int64,
                ),

            "pred":

                np.zeros(

                    (
                        N_BLOCKS,
                        len(
                            THRESHOLDS
                        ),
                    ),

                    dtype=np.int64,
                ),

            "match":

                np.zeros(

                    (
                        N_BLOCKS,
                        len(
                            THRESHOLDS
                        ),
                    ),

                    dtype=np.int64,
                ),
        }


    for ri in range(
        N_BLOCKS
    ):

        truth_block = 0


        pred_block = {

            method:

                np.zeros(
                    len(
                        THRESHOLDS
                    ),
                    dtype=np.int64,
                )

            for method in MAPS
        }


        match_block = {

            method:

                np.zeros(
                    len(
                        THRESHOLDS
                    ),
                    dtype=np.int64,
                )

            for method in MAPS
        }


        for c in range(
            3
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
                ) = smooth_masked(

                    TRUTH[
                        c,
                        ri,
                        p,
                    ],

                    mask,

                    sigma,
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
                ) = peaks(

                    truth_nu,
                    truth_valid,
                )


                keep_truth = (

                    truth_values_all
                    >= TRUTH_NU

                )


                truth_coords = (

                    truth_coords_all[
                        keep_truth
                    ]

                )


                truth_block += len(
                    truth_coords
                )


                for method, grid in MAPS.items():

                    (
                        rec_smoothed,
                        rec_valid,
                    ) = smooth_masked(

                        grid[
                            c,
                            ri,
                            p,
                        ],

                        mask,

                        sigma,
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
                    ) = peaks(

                        rec_nu,
                        valid,
                    )


                    for threshold_index, threshold in enumerate(
                        THRESHOLDS
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


                        pred_block[
                            method
                        ][
                            threshold_index
                        ] += len(
                            rec_coords
                        )


                        match_block[
                            method
                        ][
                            threshold_index
                        ] += match_count(

                            truth_coords,
                            rec_coords,
                        )


        for method in MAPS:

            BLOCK[
                scale
            ][
                method
            ][
                "truth"
            ][
                ri
            ] = truth_block


            BLOCK[
                scale
            ][
                method
            ][
                "pred"
            ][
                ri
            ] = pred_block[
                method
            ]


            BLOCK[
                scale
            ][
                method
            ][
                "match"
            ][
                ri
            ] = match_block[
                method
            ]


        print(

            f"scale={scale:.0f}'  "
            f"block={ri+1:02d}/20  "
            f"truth={truth_block}",

            flush=True,
        )


# =============================================================================
# POINT ESTIMATES + BOOTSTRAP
# =============================================================================

rng = np.random.default_rng(
    SEED
)


comparisons = [

    (
        "FUSION",
        "GAUSSIAN_STARLET",
    ),

    (
        "FUSION",
        "MCALENS",
    ),

    (
        "GAUSSIAN_STARLET",
        "MCALENS",
    ),
]


summary_rows = []


for scale in SMOOTH:


    def aggregate(
        indices,
    ):

        curves = {}


        for method in MAPS:

            truth_total = int(

                BLOCK[
                    scale
                ][
                    method
                ][
                    "truth"
                ][
                    indices
                ].sum()

            )


            predicted = (

                BLOCK[
                    scale
                ][
                    method
                ][
                    "pred"
                ][
                    indices
                ]
                .sum(
                    axis=0
                )

            )


            matched = (

                BLOCK[
                    scale
                ][
                    method
                ][
                    "match"
                ][
                    indices
                ]
                .sum(
                    axis=0
                )

            )


            curves[
                method
            ] = curve_from_counts(

                truth_total,
                predicted,
                matched,
            )


        return normalized_common_auc(
            curves
        )


    # Full 20-block point estimate.

    point = aggregate(

        np.arange(
            N_BLOCKS
        )

    )


    bootstrap_delta = {

        pair:
            np.empty(
                N_BOOT,
                dtype=np.float64,
            )

        for pair in comparisons
    }


    for iboot in range(
        N_BOOT
    ):

        indices = rng.integers(

            0,
            N_BLOCKS,

            size=N_BLOCKS,
        )


        auc = aggregate(
            indices
        )


        for pair in comparisons:

            m1, m2 = pair


            bootstrap_delta[
                pair
            ][
                iboot
            ] = (

                auc[
                    m1
                ]

                -

                auc[
                    m2
                ]

            )


    print()
    print(
        "=" * 120
    )

    print(
        f"{scale:.0f} arcmin"
    )

    print(
        "=" * 120
    )


    for method in MAPS:

        print(

            f"{method:<18} "
            f"AUC="
            f"{point[method]:.6f}"

        )


    for pair in comparisons:

        m1, m2 = pair

        values = bootstrap_delta[
            pair
        ]


        valid_values = values[
            np.isfinite(
                values
            )
        ]


        lo, hi = np.percentile(

            valid_values,

            [
                2.5,
                97.5,
            ],
        )


        delta = (

            point[
                m1
            ]

            -

            point[
                m2
            ]
        )


        fraction_gt0 = np.mean(

            valid_values
            > 0

        )


        print(

            f"{m1} - {m2}: "
            f"{delta:+.6f}  "
            f"CI95=[{lo:+.6f}, {hi:+.6f}]  "
            f"P(delta>0)="
            f"{fraction_gt0:.4f}"

        )


        summary_rows.append({

            "scale_arcmin":
                scale,

            "method_1":
                m1,

            "method_2":
                m2,

            "auc_1":
                point[
                    m1
                ],

            "auc_2":
                point[
                    m2
                ],

            "delta":
                delta,

            "ci95_low":
                lo,

            "ci95_high":
                hi,

            "bootstrap_fraction_gt0":
                fraction_gt0,

            "n_blocks":
                N_BLOCKS,

            "n_boot":
                N_BOOT,
        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    OUT
    / "peak_prauc_block_bootstrap.csv",

    index=False,
)


print()
print(
    "=" * 120
)

print(
    "BOOTSTRAP SUMMARY"
)

print(
    "=" * 120
)


print(

    summary_df.to_string(

        index=False,

        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


print()
print(
    "FILE:"
)

print(

    OUT
    / "peak_prauc_block_bootstrap.csv"

)

print()
print(
    "DONE"
)
