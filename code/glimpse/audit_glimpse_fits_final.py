#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final read-only audit of the ACTUAL frozen GLIMPSE FITS maps.

Checks:
  1. all 480 held-out maps exist
  2. identifies the reconstruction HDU/array
  3. reproduces frozen field metrics
  4. tests sign / flips / transpose / rotations
  5. reports metrics by cosmology

No retuning. No reconstruction.
"""
from project_paths import PATHS

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


# =============================================================================
# CONFIG
# =============================================================================

N = 176

DATA = Path(str(PATHS.data_root))

MASK_FILE = DATA / "WIDE12H_bin2_2arcmin_mask.npy"
KAPPA_FILE = DATA / "sampled_WIDE12H_bin2_2arcmin_kappa.npy"

REC_DIR = Path(
    str(PATHS.work_root / 'test/glimpse_recon')
)

OUT = Path(
    str(PATHS.work_root / 'test/glimpse_final_audit')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED = np.array(
    [
        0.824143,
        0.437311,
        0.564743,
    ],
    dtype=float,
)

TOL = 5e-4


# =============================================================================
# TRUTH + MASK
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

    out = np.zeros(
        mask_full.shape,
        dtype=np.float32,
    )

    out[mask_full] = np.asarray(
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
# METRICS
# =============================================================================

def metrics(
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

    r = r - r.mean()
    t = t - t.mean()

    nmse = (
        np.sum((r - t) ** 2)
        /
        np.sum(t ** 2)
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
# INSPECT FIRST FITS
# =============================================================================

FIRST = (
    REC_DIR
    / "C0_R10_P0.fits"
)

if not FIRST.exists():

    raise FileNotFoundError(
        FIRST
    )


print()
print("=" * 120)
print("FIRST FITS STRUCTURE")
print("=" * 120)

candidate_hdus = []


with fits.open(
    FIRST,
    memmap=True,
) as hdul:

    for ihdu, hdu in enumerate(
        hdul
    ):

        data = hdu.data

        if data is None:

            print(
                f"HDU {ihdu}: "
                f"name={hdu.name!r} "
                f"data=None"
            )

            continue


        arr = np.asarray(data)

        squeezed = np.squeeze(
            arr
        )


        print(
            f"HDU {ihdu}: "
            f"name={hdu.name!r} "
            f"raw_shape={arr.shape} "
            f"squeezed_shape={squeezed.shape} "
            f"dtype={arr.dtype}"
        )


        if squeezed.shape == (
            N,
            N,
        ):

            candidate_hdus.append(
                ihdu
            )


if len(candidate_hdus) == 0:

    raise RuntimeError(
        "No 176x176 image found in GLIMPSE FITS."
    )


print()
print(
    "Candidate reconstruction HDUs:",
    candidate_hdus
)


# =============================================================================
# READ A SPECIFIC HDU
# =============================================================================

def read_map(
    fn,
    ihdu,
):

    with fits.open(
        fn,
        memmap=True,
    ) as hdul:

        if ihdu >= len(hdul):

            raise RuntimeError(
                f"{fn}: missing HDU {ihdu}"
            )


        arr = np.squeeze(
            np.asarray(
                hdul[ihdu].data
            )
        )


    if arr.shape != (
        N,
        N,
    ):

        raise RuntimeError(
            f"{fn}, HDU {ihdu}: "
            f"shape={arr.shape}, "
            f"expected={(N, N)}"
        )


    if np.iscomplexobj(arr):

        print(
            f"WARNING: complex map in {fn}; "
            "using real component."
        )

        arr = arr.real


    return np.asarray(
        arr,
        dtype=np.float32,
    )


# =============================================================================
# TEST EVERY 176x176 HDU AGAINST THE FROZEN RESULT
# =============================================================================

hdu_summary = []

GRID_BY_HDU = {}


for ihdu in candidate_hdus:

    print()
    print(
        f"Loading HDU {ihdu} ..."
    )


    grid = np.empty(
        (
            3,
            20,
            8,
            N,
            N,
        ),
        dtype=np.float32,
    )


    count = 0


    for c in range(3):

        for ri, r in enumerate(
            range(10, 30)
        ):

            for p in range(8):

                fn = (
                    REC_DIR
                    / f"C{c}_R{r}_P{p}.fits"
                )


                if not fn.exists():

                    raise FileNotFoundError(
                        fn
                    )


                grid[
                    c,
                    ri,
                    p,
                ] = read_map(
                    fn,
                    ihdu,
                )


                count += 1


    if count != 480:

        raise RuntimeError(
            f"Loaded {count} maps, expected 480."
        )


    vals = []


    for c in range(3):

        for ri in range(20):

            for p in range(8):

                vals.append(
                    metrics(
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
                        MASK_PATCH[p],
                    )
                )


    mean_metrics = np.mean(
        np.asarray(vals),
        axis=0,
    )


    diff = np.abs(
        mean_metrics
        -
        EXPECTED
    )


    score = np.sum(
        diff
    )


    hdu_summary.append(
        {
            "hdu": ihdu,
            "nmse": mean_metrics[0],
            "pcc": mean_metrics[1],
            "std_ratio": mean_metrics[2],
            "abs_nmse_error": diff[0],
            "abs_pcc_error": diff[1],
            "abs_std_error": diff[2],
            "score": score,
        }
    )


    GRID_BY_HDU[
        ihdu
    ] = grid


hdu_df = pd.DataFrame(
    hdu_summary
).sort_values(
    "score"
)


print()
print("=" * 120)
print("HDU METRIC AUDIT")
print("=" * 120)

print(
    hdu_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}",
    )
)


BEST_HDU = int(
    hdu_df.iloc[0]["hdu"]
)

GLIMPSE = GRID_BY_HDU[
    BEST_HDU
]


measured = np.array(
    [
        hdu_df.iloc[0]["nmse"],
        hdu_df.iloc[0]["pcc"],
        hdu_df.iloc[0]["std_ratio"],
    ]
)


print()
print("=" * 120)
print("SELECTED GLIMPSE HDU")
print("=" * 120)

print(
    "HDU:",
    BEST_HDU
)

print(
    "Measured:",
    measured
)

print(
    "Expected:",
    EXPECTED
)

print(
    "Abs diff:",
    np.abs(
        measured
        -
        EXPECTED
    )
)


if np.any(
    np.abs(
        measured
        -
        EXPECTED
    )
    > TOL
):

    print()
    print(
        "WARNING: FITS maps do NOT reproduce "
        "the frozen GLIMPSE summary."
    )

else:

    print()
    print(
        "PASS: FITS maps reproduce the "
        "frozen GLIMPSE summary."
    )


# =============================================================================
# TRANSFORMATION AUDIT
# =============================================================================

TRANSFORMS = {

    "identity":
        lambda x: x,

    "minus":
        lambda x: -x,

    "flip_x":
        lambda x: x[:, ::-1],

    "flip_y":
        lambda x: x[::-1, :],

    "flip_xy":
        lambda x: x[::-1, ::-1],

    "transpose":
        lambda x: x.T,

    "minus_transpose":
        lambda x: -x.T,

    "rot90":
        lambda x: np.rot90(x, 1),

    "rot180":
        lambda x: np.rot90(x, 2),

    "rot270":
        lambda x: np.rot90(x, 3),
}


transform_rows = []


for name, transform in TRANSFORMS.items():

    values = []


    for c in range(3):

        for ri in range(20):

            for p in range(8):

                rec = transform(
                    GLIMPSE[
                        c,
                        ri,
                        p,
                    ]
                )


                values.append(
                    metrics(
                        rec,
                        TRUTH[
                            c,
                            ri,
                            p,
                        ],
                        MASK_PATCH[p],
                    )
                )


    values = np.asarray(
        values
    )


    transform_rows.append(
        {
            "transform": name,
            "nmse": values[:, 0].mean(),
            "pcc": values[:, 1].mean(),
            "std_ratio": values[:, 2].mean(),
        }
    )


transform_df = pd.DataFrame(
    transform_rows
).sort_values(
    "nmse"
)


print()
print("=" * 120)
print("SIGN / ORIENTATION AUDIT")
print("=" * 120)

print(
    transform_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.9f}",
    )
)


BEST_TRANSFORM = (
    transform_df.iloc[0][
        "transform"
    ]
)


print()
print(
    "Best transform:",
    BEST_TRANSFORM
)


if BEST_TRANSFORM == "identity":

    print(
        "PASS: no trivial sign/orientation "
        "change improves GLIMPSE."
    )

else:

    print(
        "WARNING: investigate GLIMPSE orientation/sign."
    )


# =============================================================================
# BY COSMOLOGY
# =============================================================================

rows = []


for c in range(3):

    for ri, r in enumerate(
        range(10, 30)
    ):

        for p in range(8):

            nmse, pcc, std = metrics(
                GLIMPSE[
                    c,
                    ri,
                    p,
                ],
                TRUTH[
                    c,
                    ri,
                    p,
                ],
                MASK_PATCH[p],
            )


            rows.append(
                {
                    "c": c,
                    "r": r,
                    "p": p,
                    "nmse": nmse,
                    "pcc": pcc,
                    "std_ratio": std,
                }
            )


case_df = pd.DataFrame(
    rows
)


print()
print("=" * 120)
print("GLIMPSE BY COSMOLOGY")
print("=" * 120)

print(
    case_df
    .groupby("c")[
        [
            "nmse",
            "pcc",
            "std_ratio",
        ]
    ]
    .mean()
    .to_string(
        float_format=lambda x: f"{x:.9f}"
    )
)


# =============================================================================
# SAVE
# =============================================================================

hdu_df.to_csv(
    OUT
    / "glimpse_hdu_audit.csv",
    index=False,
)

transform_df.to_csv(
    OUT
    / "glimpse_transform_audit.csv",
    index=False,
)

case_df.to_csv(
    OUT
    / "glimpse_case_metrics.csv",
    index=False,
)


print()
print("=" * 120)
print("FILES")
print("=" * 120)

for fn in [
    "glimpse_hdu_audit.csv",
    "glimpse_transform_audit.csv",
    "glimpse_case_metrics.csv",
]:

    print(
        OUT
        / fn
    )


print()
print("DONE")
