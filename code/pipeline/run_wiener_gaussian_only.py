#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gaussian-only / Wiener-like baseline.

Objective
---------
min_x  1/2 || M(Ax-y) ||_2^2
     + beta/2 sum_{k:P(k)>0} |Fx(k)|^2 / P(k)

The Fourier support P(k)>0 is enforced exactly.

Validation:
    R0-R9, 3 cosmologies, 8 patches = 240 patches

Test:
    R10-R29, 3 cosmologies, 8 patches = 480 patches

beta is selected ONLY on validation.
"""
from project_paths import PATHS

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(
    0,
    str(PATHS.code_root)
)

from weak_lensing_pipeline import WeakLensingOperator


# =============================================================================
# CONFIG
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

N = 176

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

PK_FILE = Path(
    str(PATHS.prior_file)
)

VAL_DATA_ROOT = Path(
    str(PATHS.work_root / 'validation/fair_npz')
)

TEST_DATA_ROOT = Path(
    str(PATHS.work_root / 'test/fair_npz')
)

OUT = Path(
    str(PATHS.work_root / 'wiener_gaussian_only')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BETA_GRID = [
    2.5e-4,
    5.0e-4,
    1.0e-3,
    2.0e-3,
    4.0e-3,
]

CG_MAXITER = 300
CG_RTOL = 1.0e-7

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


# =============================================================================
# CHECKS
# =============================================================================

for fn in [
    MASK_FILE,
    PK_FILE,
]:
    if not fn.exists():
        raise FileNotFoundError(fn)

for root in [
    VAL_DATA_ROOT,
    TEST_DATA_ROOT,
]:
    if not root.exists():
        raise FileNotFoundError(
            f"{root}\n"
            "Need the frozen fair_npz products used by the "
            "existing Gaussian+Starlet benchmark."
        )


# =============================================================================
# OPERATOR ADAPTERS
# =============================================================================

op = WeakLensingOperator(
    N,
    N,
    DEVICE,
)


def A(x):
    return op.forward(x)


def AH(y):
    z = op.adjoint(y)
    return z.real if torch.is_complex(z) else z


# =============================================================================
# P_KAPPA MAP
# =============================================================================

def make_pk_map(
    pk,
    H,
    W,
):
    pk = np.asarray(
        pk,
        dtype=np.float64,
    ).copy()

    pk[0] = 0.0

    ext = np.concatenate(
        [
            pk,
            np.zeros(
                1,
                dtype=np.float64,
            ),
        ]
    )

    npk = len(pk)

    ky = np.arange(H) - H / 2.0
    kx = np.arange(W) - W / 2.0

    KY, KX = np.meshgrid(
        ky,
        kx,
        indexing="ij",
    )

    ir = np.sqrt(
        KX**2 + KY**2
    ).astype(np.int64)

    ir[
        ir > npk
    ] = npk

    return np.fft.ifftshift(
        ext[ir]
    )


pk1d = np.load(
    PK_FILE
).astype(np.float64)

pk_np = make_pk_map(
    pk1d,
    N,
    N,
)

PK = torch.as_tensor(
    pk_np,
    dtype=torch.float32,
    device=DEVICE,
)

SUPPORT = (
    PK > 0
).float()

INVP = torch.where(
    PK > 0,
    1.0 / PK,
    torch.zeros_like(PK),
)


# =============================================================================
# SUPPORT PROJECTION / PRIOR
# =============================================================================

def support_project(x):
    X = torch.fft.fft2(
        x,
        norm="ortho",
    )

    return torch.fft.ifft2(
        X * SUPPORT,
        norm="ortho",
    ).real


def prior_precision(x):
    X = torch.fft.fft2(
        x,
        norm="ortho",
    )

    return torch.fft.ifft2(
        X * INVP,
        norm="ortho",
    ).real


# =============================================================================
# CG
# =============================================================================

@torch.no_grad()
def cg_solve(
    y,
    mask,
    beta,
    x0=None,
):
    """
    Solve on the Fourier support P(k)>0:

        S [A^H M A + beta C^-1] S x
        =
        S A^H y
    """

    rhs = support_project(
        AH(
            mask * y
        )
    )

    if x0 is None:
        x = torch.zeros_like(
            rhs
        )
    else:
        x = support_project(
            x0.clone()
        )

    def H(v):
        v = support_project(v)

        data = AH(
            mask * A(v)
        )

        prior = beta * prior_precision(
            v
        )

        return support_project(
            data + prior
        )

    r = rhs - H(x)
    p = r.clone()

    rr = torch.sum(
        r * r
    )

    rhs_norm = max(
        float(
            torch.linalg.norm(rhs).item()
        ),
        1.0e-30,
    )

    rel = (
        float(
            torch.sqrt(rr).item()
        )
        / rhs_norm
    )

    nit = 0

    for it in range(
        1,
        CG_MAXITER + 1,
    ):
        Hp = H(p)

        denom = torch.sum(
            p * Hp
        )

        alpha = (
            rr
            / torch.clamp(
                denom,
                min=1.0e-30,
            )
        )

        x = x + alpha * p
        r = r - alpha * Hp

        rr_new = torch.sum(
            r * r
        )

        rel = (
            float(
                torch.sqrt(
                    torch.clamp(
                        rr_new,
                        min=0.0,
                    )
                ).item()
            )
            / rhs_norm
        )

        nit = it

        if rel < CG_RTOL:
            rr = rr_new
            break

        cg_beta = (
            rr_new
            / torch.clamp(
                rr,
                min=1.0e-30,
            )
        )

        p = r + cg_beta * p
        rr = rr_new

    x = support_project(x)

    return x, nit, rel


# =============================================================================
# METRICS
# =============================================================================

def metrics(
    rec,
    truth,
    mask,
):
    m = mask.astype(bool)

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[m]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[m]

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
        np.dot(r, t)
        /
        (
            np.linalg.norm(r)
            * np.linalg.norm(t)
            + 1.0e-30
        )
    )

    std = (
        np.std(r)
        /
        (
            np.std(t)
            + 1.0e-30
        )
    )

    return (
        float(nmse),
        float(pcc),
        float(std),
    )


# =============================================================================
# LOAD FULL MAP
# =============================================================================

mask_full = np.load(
    MASK_FILE
).astype(bool)


def load_case(
    root,
    c,
    r,
    p,
):
    fn = (
        root
        / f"C{c}_R{r}.npz"
    )

    if not fn.exists():
        raise FileNotFoundError(fn)

    with np.load(fn) as z:
        if "truth" not in z:
            raise KeyError(
                f"{fn}: missing 'truth'"
            )

        if "gamma" not in z:
            raise KeyError(
                f"{fn}: missing 'gamma'"
            )

        truth_full = np.asarray(
            z["truth"],
            dtype=np.float32,
        )

        gamma_full = np.asarray(
            z["gamma"],
            dtype=np.complex64,
        )

    r0, r1 = PATCH_ROWS[p]

    truth = truth_full[
        r0:r1,
        :
    ].copy()

    gamma = gamma_full[
        r0:r1,
        :
    ].copy()

    mask = mask_full[
        r0:r1,
        :
    ].copy()

    gamma *= mask

    return (
        truth,
        gamma,
        mask,
    )


# =============================================================================
# VALIDATION
# =============================================================================

print()
print("=" * 120)
print("WIENER / GAUSSIAN-ONLY VALIDATION")
print("=" * 120)

val_rows = []

for beta in BETA_GRID:
    print()
    print(
        f"beta = {beta:.6g}"
    )

    t0 = time.perf_counter()

    rows_beta = []

    for c in range(3):
        for r in range(10):
            for p in range(8):

                truth, gamma, mask = (
                    load_case(
                        VAL_DATA_ROOT,
                        c,
                        r,
                        p,
                    )
                )

                y = torch.as_tensor(
                    gamma,
                    dtype=torch.complex64,
                    device=DEVICE,
                )

                mt = torch.as_tensor(
                    mask.astype(np.float32),
                    dtype=torch.float32,
                    device=DEVICE,
                )

                rec_t, nit, rel = cg_solve(
                    y=y,
                    mask=mt,
                    beta=beta,
                )

                rec = (
                    rec_t
                    .cpu()
                    .numpy()
                )

                nmse, pcc, std = metrics(
                    rec,
                    truth,
                    mask,
                )

                row = {
                    "beta": beta,
                    "c": c,
                    "r": r,
                    "p": p,
                    "nmse": nmse,
                    "pcc": pcc,
                    "std_ratio": std,
                    "cg_iterations": nit,
                    "cg_rel": rel,
                }

                val_rows.append(row)
                rows_beta.append(row)

    dfb = pd.DataFrame(
        rows_beta
    )

    print(
        f"  NMSE = "
        f"{dfb.nmse.mean():.9f}"
    )

    print(
        f"  PCC  = "
        f"{dfb.pcc.mean():.9f}"
    )

    print(
        f"  STD  = "
        f"{dfb.std_ratio.mean():.9f}"
    )

    print(
        f"  CG   = "
        f"{dfb.cg_iterations.mean():.1f} iterations"
    )

    print(
        f"  time = "
        f"{(time.perf_counter()-t0)/60:.2f} min"
    )


val_df = pd.DataFrame(
    val_rows
)

val_df.to_csv(
    OUT
    / "wiener_validation_cases.csv",
    index=False,
)


val_summary = (
    val_df
    .groupby("beta")
    .agg(
        nmse=("nmse", "mean"),
        pcc=("pcc", "mean"),
        std_ratio=("std_ratio", "mean"),
        cg_iterations=("cg_iterations", "mean"),
    )
    .reset_index()
)


best_row = (
    val_summary
    .sort_values(
        "nmse",
        ascending=True,
    )
    .iloc[0]
)


BETA_BEST = float(
    best_row.beta
)


val_summary[
    "selected"
] = (
    val_summary.beta
    == BETA_BEST
)


val_summary.to_csv(
    OUT
    / "wiener_validation_grid.csv",
    index=False,
)


print()
print("=" * 120)
print("FROZEN WIENER BETA")
print("=" * 120)

print(
    val_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.9g}",
    )
)

print()
print(
    "BETA_BEST =",
    BETA_BEST
)


# =============================================================================
# HELD-OUT TEST
# =============================================================================

print()
print("=" * 120)
print("WIENER / GAUSSIAN-ONLY HELD-OUT TEST")
print("=" * 120)


test_maps = np.empty(
    (
        3,
        20,
        8,
        N,
        N,
    ),
    dtype=np.float32,
)

test_rows = []


for c in range(3):
    for ri, r in enumerate(
        range(10, 30)
    ):
        for p in range(8):

            truth, gamma, mask = (
                load_case(
                    TEST_DATA_ROOT,
                    c,
                    r,
                    p,
                )
            )

            y = torch.as_tensor(
                gamma,
                dtype=torch.complex64,
                device=DEVICE,
            )

            mt = torch.as_tensor(
                mask.astype(np.float32),
                dtype=torch.float32,
                device=DEVICE,
            )

            rec_t, nit, rel = cg_solve(
                y=y,
                mask=mt,
                beta=BETA_BEST,
            )

            rec = (
                rec_t
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            test_maps[
                c,
                ri,
                p,
            ] = rec

            nmse, pcc, std = metrics(
                rec,
                truth,
                mask,
            )

            test_rows.append({
                "c": c,
                "r": r,
                "p": p,
                "beta": BETA_BEST,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": std,
                "cg_iterations": nit,
                "cg_rel": rel,
            })


np.save(
    OUT
    / "wiener_test_reconstruction.npy",
    test_maps,
)


test_df = pd.DataFrame(
    test_rows
)

test_df.to_csv(
    OUT
    / "wiener_test_cases.csv",
    index=False,
)


print()
print(
    f"TEST NMSE = "
    f"{test_df.nmse.mean():.9f}"
)

print(
    f"TEST PCC  = "
    f"{test_df.pcc.mean():.9f}"
)

print(
    f"TEST STD  = "
    f"{test_df.std_ratio.mean():.9f}"
)


print()
print("=" * 120)
print("BY COSMOLOGY")
print("=" * 120)

print(
    test_df
    .groupby("c")
    [["nmse", "pcc", "std_ratio"]]
    .mean()
    .to_string(
        float_format=lambda x: f"{x:.9f}"
    )
)


print()
print("=" * 120)
print("FILES")
print("=" * 120)

for fn in [
    "wiener_validation_cases.csv",
    "wiener_validation_grid.csv",
    "wiener_test_cases.csv",
    "wiener_test_reconstruction.npy",
]:
    print(
        OUT / fn
    )

print()
print("DONE")
