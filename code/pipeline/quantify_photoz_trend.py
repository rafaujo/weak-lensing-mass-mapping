#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantify dependence of reconstruction NMSE on FAIR Delta_z.

Important
---------
Patches are NOT treated as independent.

For each:
    cosmology x realization

we first aggregate the 8 patch metrics.

Then fit:

    NMSE = intercept + slope * Delta_z

The bootstrap resamples the 20 realization indices while
preserving the three cosmologies as paired observations.

MCALens is loaded explicitly from NPZ key:

    "mcalens"
"""
from project_paths import PATHS

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import linregress


# =============================================================================
# CONFIG
# =============================================================================

N = 176

N_BOOT = 10000

SEED = 20260930

MCALENS_KEY = "mcalens"


# =============================================================================
# PATHS
# =============================================================================

DATA = Path(
    str(PATHS.data_root)
)

LABEL_FILE = (
    DATA
    / "sampled_label.npy"
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
    str(PATHS.work_root / 'test/photoz_audit')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# INPUT CHECKS
# =============================================================================

for fn in [
    LABEL_FILE,
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
# LOAD FAIR
# =============================================================================

labels = np.load(
    LABEL_FILE
)

mask_full = np.load(
    MASK_FILE
).astype(
    bool
)

kappa_vec = np.load(
    KAPPA_FILE,
    mmap_mode="r",
)


print()
print(
    "=" * 120
)

print(
    "PHOTO-Z AUDIT"
)

print(
    "=" * 120
)

print(
    "labels shape:",
    labels.shape
)


if labels.shape[-1] < 5:

    raise RuntimeError(

        "Expected FAIR label ordering:\n"
        "[Omega_m, S8, T_AGN, f0, Delta_z]"

    )


# =============================================================================
# PATCHES
# =============================================================================

PATCH_ROWS = [

    (
        p * N,
        (p + 1) * N,
    )

    for p in range(
        8
    )
]


MASKS = np.stack(

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

def full_truth(
    vec,
):

    x = np.zeros(
        mask_full.shape,
        dtype=np.float32,
    )

    x[
        mask_full
    ] = np.asarray(
        vec,
        dtype=np.float32,
    )

    return x


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
        range(
            10,
            30,
        )
    ):

        full = full_truth(

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
# CANONICALIZE
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
        f"Unsupported shape: {arr.shape}"
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
# LOAD MCALENS — EXPLICIT "mcalens" KEY
# =============================================================================

def load_mcalens():

    x = np.empty_like(
        GS
    )


    schema_printed = False


    for c in range(
        3
    ):

        for ri, r in enumerate(
            range(
                10,
                30,
            )
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

                    if not schema_printed:

                        print()
                        print(
                            "MCALens first-file keys:"
                        )

                        print(
                            z.files
                        )

                        schema_printed = True


                    if MCALENS_KEY not in z:

                        raise KeyError(

                            f"{fn}: expected key "
                            f"{MCALENS_KEY!r}; "
                            f"available={z.files}"

                        )


                    arr = np.asarray(

                        z[
                            MCALENS_KEY
                        ],

                        dtype=np.float32,
                    )


                if arr.shape != (
                    N,
                    N,
                ):

                    raise ValueError(

                        f"{fn}: key "
                        f"{MCALENS_KEY!r} "
                        f"has shape {arr.shape}"

                    )


                x[
                    c,
                    ri,
                    p,
                ] = arr


    return x


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
# METRICS
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
            (r - t)**2
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
        float(
            nmse
        ),
        float(
            pcc
        ),
        float(
            std
        ),
    )


def all_patch_metrics(
    grid,
):

    values = []


    for c in range(
        3
    ):

        for ri in range(
            20
        ):

            for p in range(
                8
            ):

                values.append(

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

                        MASKS[
                            p
                        ],
                    )
                )


    return np.mean(
        np.asarray(
            values
        ),
        axis=0,
    )


# =============================================================================
# SANITY CHECK
# =============================================================================

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
    "FROZEN MAP SANITY"
)

print(
    "=" * 120
)


for method, grid in MAPS.items():

    measured = all_patch_metrics(
        grid
    )


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
        EXPECTED[
            method
        ]
    )


    if np.any(

        np.abs(

            measured

            -

            EXPECTED[
                method
            ]

        )

        > 5e-4

    ):

        raise RuntimeError(

            f"{method}: frozen-map sanity failed."

        )


print()
print(
    "All sanity checks PASSED."
)


# =============================================================================
# REALIZATION-COSMOLOGY BLOCK TABLE
# =============================================================================

rows = []


for method, grid in MAPS.items():

    for c in range(
        3
    ):

        for ri, r in enumerate(
            range(
                10,
                30,
            )
        ):

            patch_nmse = []
            patch_pcc = []
            patch_std = []
            weights = []


            for p in range(
                8
            ):

                mask = MASKS[
                    p
                ]


                (
                    nmse,
                    pcc,
                    std,
                ) = field_metrics(

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

                    mask,
                )


                patch_nmse.append(
                    nmse
                )

                patch_pcc.append(
                    pcc
                )

                patch_std.append(
                    std
                )

                weights.append(
                    mask.sum()
                )


            # FAIR label order:
            #
            # [Omega_m, S8, T_AGN, f0, Delta_z]

            delta_z = float(

                labels[
                    c,
                    r,
                    4
                ]

            )


            rows.append({

                "method":
                    method,

                "c":
                    c,

                "r":
                    r,

                "delta_z":
                    delta_z,

                "nmse":

                    np.average(
                        patch_nmse,
                        weights=weights,
                    ),

                "pcc":

                    np.average(
                        patch_pcc,
                        weights=weights,
                    ),

                "std_ratio":

                    np.average(
                        patch_std,
                        weights=weights,
                    ),
            })


df = pd.DataFrame(
    rows
)


df.to_csv(

    OUT
    / "photoz_block_metrics.csv",

    index=False,
)


# =============================================================================
# CHECK PAIRED DELTA_Z ACROSS COSMOLOGIES
# =============================================================================

print()
print(
    "=" * 120
)

print(
    "DELTA_Z PAIRING CHECK"
)

print(
    "=" * 120
)


pair_table = (

    df[
        df.method
        == "GAUSSIAN_STARLET"
    ]

    .pivot(
        index="r",
        columns="c",
        values="delta_z",
    )
)


print(
    pair_table.head(
        20
    ).to_string()
)


max_pair_diff = (

    pair_table
    .sub(
        pair_table[
            0
        ],
        axis=0,
    )
    .abs()
    .to_numpy()
    .max()
)


print()
print(
    "Max Delta_z difference across cosmologies "
    "for paired realization index:",
    max_pair_diff
)


# =============================================================================
# REGRESSION + REALIZATION-BLOCK BOOTSTRAP
# =============================================================================

rng = np.random.default_rng(
    SEED
)


summary_rows = []


for method in MAPS:

    d = df[
        df.method
        == method
    ].copy()


    # ---------------------------------------------------------
    # Raw balanced regression.
    # Cosmology is balanced at each Delta_z realization.
    # ---------------------------------------------------------

    fit = linregress(

        d[
            "delta_z"
        ],

        d[
            "nmse"
        ],
    )


    # ---------------------------------------------------------
    # Also remove each cosmology's mean NMSE.
    #
    # Since Delta_z is paired across cosmologies this should
    # give essentially the same slope, but it makes the
    # interpretation explicit.
    # ---------------------------------------------------------

    d[
        "nmse_demeaned_cosmo"
    ] = (

        d[
            "nmse"
        ]

        -

        d.groupby(
            "c"
        )[
            "nmse"
        ].transform(
            "mean"
        )
    )


    fit_demeaned = linregress(

        d[
            "delta_z"
        ],

        d[
            "nmse_demeaned_cosmo"
        ],
    )


    slopes = np.empty(
        N_BOOT,
        dtype=np.float64,
    )


    slopes_demeaned = np.empty(
        N_BOOT,
        dtype=np.float64,
    )


    realization_ids = np.arange(
        10,
        30
    )


    for iboot in range(
        N_BOOT
    ):

        sampled = rng.choice(

            realization_ids,

            size=len(
                realization_ids
            ),

            replace=True,
        )


        pieces = []


        for boot_index, original_r in enumerate(
            sampled
        ):

            tmp = d[
                d.r
                == original_r
            ].copy()


            tmp[
                "bootstrap_block"
            ] = boot_index


            pieces.append(
                tmp
            )


        boot_df = pd.concat(

            pieces,

            ignore_index=True,
        )


        boot_fit = linregress(

            boot_df[
                "delta_z"
            ],

            boot_df[
                "nmse"
            ],
        )


        boot_df[
            "nmse_demeaned_cosmo"
        ] = (

            boot_df[
                "nmse"
            ]

            -

            boot_df.groupby(
                "c"
            )[
                "nmse"
            ].transform(
                "mean"
            )
        )


        boot_fit_dm = linregress(

            boot_df[
                "delta_z"
            ],

            boot_df[
                "nmse_demeaned_cosmo"
            ],
        )


        slopes[
            iboot
        ] = boot_fit.slope


        slopes_demeaned[
            iboot
        ] = boot_fit_dm.slope


    ci_low, ci_high = np.percentile(

        slopes,

        [
            2.5,
            97.5,
        ],
    )


    dm_low, dm_high = np.percentile(

        slopes_demeaned,

        [
            2.5,
            97.5,
        ],
    )


    sigma_z = 0.022


    summary_rows.append({

        "method":
            method,

        "mean_nmse":
            d.nmse.mean(),

        "slope_nmse_per_delta_z":
            fit.slope,

        "slope_ci95_low":
            ci_low,

        "slope_ci95_high":
            ci_high,

        "pearson_r":
            fit.rvalue,

        "effect_delta_z_plus_0p022":
            fit.slope
            * sigma_z,

        "relative_effect_0p022_percent":

            100.0
            * fit.slope
            * sigma_z
            / d.nmse.mean(),

        "demeaned_slope":
            fit_demeaned.slope,

        "demeaned_ci95_low":
            dm_low,

        "demeaned_ci95_high":
            dm_high,

        "n_realization_cosmo_blocks":
            len(d),

        "n_independent_realization_ids":
            20,

        "n_boot":
            N_BOOT,
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    OUT
    / "photoz_regression_summary.csv",

    index=False,
)


# =============================================================================
# REPORT
# =============================================================================

print()
print(
    "=" * 170
)

print(
    "PHOTO-Z TREND SUMMARY"
)

print(
    "=" * 170
)


print(

    summary_df.to_string(

        index=False,

        float_format=lambda x: (
            f"{x:.8e}"
        ),
    )
)


print()
print(
    "Interpretation:"
)

print(
    "  - If the bootstrap CI95 of the slope contains zero, "
    "the representative sample does not resolve a monotonic "
    "Delta_z dependence."
)

print(
    "  - The demeaned-cosmology slope is included as a "
    "cross-check against the different baseline NMSE of C0/C1/C2."
)


print()
print(
    "FILES:"
)

print(

    OUT
    / "photoz_block_metrics.csv"

)

print(

    OUT
    / "photoz_regression_summary.csv"

)

print()
print(
    "DONE"
)
