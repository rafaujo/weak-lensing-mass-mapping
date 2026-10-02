#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paired realization-block bootstrap:
Gaussian+Starlet versus Gaussian-only/Wiener.

Held-out test:
    C0-C2
    R10-R29
    P0-P7

Independent bootstrap unit:
    realization index R

Each block therefore retains:
    3 cosmologies x 8 patches = 24 paired cases.

Metrics:
    NMSE
    PCC
    std ratio

Also reports relative NMSE reduction:
    (NMSE_Wiener - NMSE_GS) / NMSE_Wiener

No tuning is performed.
"""
from project_paths import PATHS

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

N = 176
N_BOOT = 10000
SEED = 20260930

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

WIENER_CSV = Path(
    str(PATHS.work_root / 'wiener_gaussian_only/wiener_test_cases.csv')
)

OUT = Path(
    str(PATHS.work_root / 'test/wiener_vs_gaussian_starlet_bootstrap')
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
    WIENER_CSV,
]:
    if not fn.exists():
        raise FileNotFoundError(fn)


# =============================================================================
# LOAD MASK / TRUTH
# =============================================================================

mask_full = np.load(
    MASK_FILE
).astype(bool)

kappa_vec = np.load(
    KAPPA_FILE,
    mmap_mode="r",
)

PATCH_ROWS = [
    (
        p * N,
        (p + 1) * N,
    )
    for p in range(8)
]

MASK_PATCH = np.stack(
    [
        mask_full[r0:r1, :]
        for r0, r1 in PATCH_ROWS
    ],
    axis=0,
)


def vector_to_full(vec):

    full = np.zeros(
        mask_full.shape,
        dtype=np.float32,
    )

    full[mask_full] = np.asarray(
        vec,
        dtype=np.float32,
    )

    return full


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


for c in range(3):

    for ri, r in enumerate(
        range(10, 30)
    ):

        full = vector_to_full(
            kappa_vec[c, r]
        )

        for p, (r0, r1) in enumerate(
            PATCH_ROWS
        ):

            TRUTH[
                c,
                ri,
                p,
            ] = full[r0:r1, :]


# =============================================================================
# LOAD G+S
# =============================================================================

GS = np.load(
    GS_FILE,
    mmap_mode="r",
)

GS = np.asarray(GS)


if GS.shape == (
    3,
    30,
    8,
    N,
    N,
):

    GS = GS[
        :,
        10:30,
        :,
        :,
        :,
    ]


elif GS.shape == (
    480,
    N,
    N,
):

    GS = GS.reshape(
        3,
        20,
        8,
        N,
        N,
    )


if GS.shape != (
    3,
    20,
    8,
    N,
    N,
):

    raise ValueError(
        f"Unexpected G+S shape: {GS.shape}"
    )


# =============================================================================
# METRIC
# =============================================================================

def field_metrics(
    rec,
    truth,
    mask,
):

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[mask]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[mask]

    r -= r.mean()
    t -= t.mean()

    nmse = (
        np.sum(
            (r - t) ** 2
        )
        /
        np.sum(
            t ** 2
        )
    )

    pcc = (
        np.dot(r, t)
        /
        (
            np.linalg.norm(r)
            *
            np.linalg.norm(t)
            +
            1e-30
        )
    )

    std_ratio = (
        np.std(r)
        /
        (
            np.std(t)
            +
            1e-30
        )
    )

    return (
        float(nmse),
        float(pcc),
        float(std_ratio),
    )


# =============================================================================
# BUILD G+S CASE TABLE
# =============================================================================

gs_rows = []


for c in range(3):

    for ri, r in enumerate(
        range(10, 30)
    ):

        for p in range(8):

            nmse, pcc, std = field_metrics(
                GS[c, ri, p],
                TRUTH[c, ri, p],
                MASK_PATCH[p],
            )

            gs_rows.append(
                {
                    "c": c,
                    "r": r,
                    "p": p,
                    "gs_nmse": nmse,
                    "gs_pcc": pcc,
                    "gs_std_ratio": std,
                }
            )


gs_df = pd.DataFrame(
    gs_rows
)


# =============================================================================
# LOAD WIENER
# =============================================================================

w_df = pd.read_csv(
    WIENER_CSV
)


required = {
    "c",
    "r",
    "p",
    "nmse",
    "pcc",
    "std_ratio",
}


missing = (
    required
    -
    set(w_df.columns)
)


if missing:

    raise RuntimeError(
        f"Wiener CSV missing columns: {sorted(missing)}"
    )


w_df = (
    w_df[
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
            "std_ratio",
        ]
    ]
    .rename(
        columns={
            "nmse":
                "wiener_nmse",

            "pcc":
                "wiener_pcc",

            "std_ratio":
                "wiener_std_ratio",
        }
    )
)


# =============================================================================
# MERGE — EXACT PAIRING
# =============================================================================

df = gs_df.merge(
    w_df,
    on=[
        "c",
        "r",
        "p",
    ],
    how="inner",
    validate="one_to_one",
)


if len(df) != 480:

    raise RuntimeError(
        f"Expected 480 paired cases; got {len(df)}"
    )


# =============================================================================
# FROZEN SANITY
# =============================================================================

expected = {
    "gs_nmse":
        0.642805480,

    "gs_pcc":
        0.59660041,

    "gs_std_ratio":
        0.58345016,

    "wiener_nmse":
        0.686769155,

    "wiener_pcc":
        0.568533165,

    "wiener_std_ratio":
        0.471536857,
}


print()
print("=" * 120)
print("FROZEN SANITY CHECK")
print("=" * 120)


for col, ref in expected.items():

    value = df[col].mean()

    diff = abs(
        value
        -
        ref
    )

    print(
        f"{col:<24} "
        f"{value:.9f} "
        f"expected={ref:.9f} "
        f"diff={diff:.3e}"
    )

    if diff > 5e-4:

        raise RuntimeError(
            f"Sanity failed for {col}"
        )


print()
print(
    "All frozen metric checks PASSED."
)


# =============================================================================
# POINT ESTIMATES
# =============================================================================

def summarize(
    d,
):

    gs_nmse = d[
        "gs_nmse"
    ].mean()

    w_nmse = d[
        "wiener_nmse"
    ].mean()

    gs_pcc = d[
        "gs_pcc"
    ].mean()

    w_pcc = d[
        "wiener_pcc"
    ].mean()

    gs_std = d[
        "gs_std_ratio"
    ].mean()

    w_std = d[
        "wiener_std_ratio"
    ].mean()

    return {
        "delta_nmse_gs_minus_wiener":
            gs_nmse
            -
            w_nmse,

        "relative_nmse_reduction_gs":
            (
                w_nmse
                -
                gs_nmse
            )
            /
            w_nmse,

        "delta_pcc_gs_minus_wiener":
            gs_pcc
            -
            w_pcc,

        "delta_std_gs_minus_wiener":
            gs_std
            -
            w_std,
    }


point = summarize(
    df
)


# =============================================================================
# BLOCK BOOTSTRAP
# =============================================================================

rng = np.random.default_rng(
    SEED
)

REALIZATIONS = np.arange(
    10,
    30,
)

boot = {
    key:
        np.empty(
            N_BOOT,
            dtype=np.float64,
        )
    for key in point
}


for ib in range(
    N_BOOT
):

    sampled_r = rng.choice(
        REALIZATIONS,
        size=len(
            REALIZATIONS
        ),
        replace=True,
    )

    pieces = []

    for boot_id, r in enumerate(
        sampled_r
    ):

        tmp = df[
            df["r"]
            == r
        ].copy()

        tmp[
            "bootstrap_block"
        ] = boot_id

        pieces.append(
            tmp
        )


    b = pd.concat(
        pieces,
        ignore_index=True,
    )

    s = summarize(
        b
    )

    for key in boot:

        boot[
            key
        ][ib] = s[
            key
        ]


# =============================================================================
# OUTPUT
# =============================================================================

rows = []


for key, estimate in point.items():

    values = boot[
        key
    ]

    lo, hi = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    rows.append(
        {
            "quantity":
                key,

            "point":
                estimate,

            "ci95_low":
                lo,

            "ci95_high":
                hi,

            "bootstrap_fraction_gt0":
                np.mean(
                    values > 0
                ),

            "n_blocks":
                20,

            "n_cases":
                480,

            "n_boot":
                N_BOOT,
        }
    )


summary = pd.DataFrame(
    rows
)


df.to_csv(
    OUT
    / "wiener_vs_gs_paired_cases.csv",
    index=False,
)


summary.to_csv(
    OUT
    / "wiener_vs_gs_block_bootstrap.csv",
    index=False,
)


print()
print("=" * 150)
print("WIENER vs GAUSSIAN+STARLET")
print("=" * 150)


print(
    f"Wiener NMSE = "
    f"{df.wiener_nmse.mean():.9f}"
)

print(
    f"G+S NMSE    = "
    f"{df.gs_nmse.mean():.9f}"
)

print(
    f"Wiener PCC  = "
    f"{df.wiener_pcc.mean():.9f}"
)

print(
    f"G+S PCC     = "
    f"{df.gs_pcc.mean():.9f}"
)

print(
    f"Wiener STD  = "
    f"{df.wiener_std_ratio.mean():.9f}"
)

print(
    f"G+S STD     = "
    f"{df.gs_std_ratio.mean():.9f}"
)


print()
print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}",
    )
)


print()
print("FILES:")

print(
    OUT
    / "wiener_vs_gs_paired_cases.csv"
)

print(
    OUT
    / "wiener_vs_gs_block_bootstrap.csv"
)

print()
print("DONE")
