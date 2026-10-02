from project_paths import PATHS
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(PATHS.code_root))

from weak_lensing_pipeline import (
    WeakLensingOperator,
    KaiserSquires,
)


# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE = Path(
    str(PATHS.work_root / 'validation/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

PK_FILE = Path(
    str(PATHS.prior_file)
)

OUT = Path(
    str(PATHS.work_root / 'validation/starlet_gaussian_prior')
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# Existing lambda=0.05 results.
OLD_CSV = OUT / "gaussian_starlet_validation.csv"

# New combined grid.
CSV = OUT / "gaussian_starlet_joint_validation.csv"

SUMMARY_CSV = (
    OUT
    / "gaussian_starlet_joint_validation_summary.csv"
)

BYCOSMO_CSV = (
    OUT
    / "gaussian_starlet_joint_validation_by_cosmology.csv"
)

N = 176
J = 4

# ------------------------------------------------------------
# JOINT VALIDATION GRID
# ------------------------------------------------------------
LAMBDA_GRID = [
    0.03,
    0.05,
    0.07,
]

BETA_GRID = [
    6.0e-4,
    1.0e-3,
    2.0e-3,
]

ALPHA = np.array(
    [
        1.0,
        0.225148,
        0.096029,
        0.046286,
    ],
    dtype=np.float32,
)

NITER = 6000

TAU = 0.50
SIGMA = 0.50
THETA = 1.0

BATCH = 16

COSMOS = range(3)
REALS = range(10)
PATCHES = range(8)

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


# ============================================================
# BATCH ADAPTERS
# ============================================================
def op_forward_batch(op, x):
    """
    Canonical operator acts on one 2-D map.
    vmap applies it independently to the batch.
    """
    if x.ndim == 2:
        return op.forward(x)

    if x.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), got {tuple(x.shape)}"
        )

    return torch.vmap(
        op.forward
    )(x)


def op_adjoint_batch(op, y):
    if y.ndim == 2:
        return op.adjoint(y)

    if y.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), got {tuple(y.shape)}"
        )

    return torch.vmap(
        op.adjoint
    )(y)


def ks_reconstruct_batch(
    ks,
    y,
    smooth_sigma,
):
    if y.ndim == 2:
        return ks.reconstruct(
            y,
            smooth_sigma=smooth_sigma,
        )

    if y.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), got {tuple(y.shape)}"
        )

    return torch.vmap(
        lambda yy: ks.reconstruct(
            yy,
            smooth_sigma=smooth_sigma,
        )
    )(y)


# ============================================================
# METRICS
# ============================================================
def ms_metrics_batch(
    rec,
    truth,
    masks,
):
    """
    Mass-sheet-invariant metrics on observed pixels.
    """
    out = []

    for i in range(
        rec.shape[0]
    ):
        m = masks[i].astype(bool)

        r = np.asarray(
            rec[i],
            dtype=np.float64,
        )[m]

        t = np.asarray(
            truth[i],
            dtype=np.float64,
        )[m]

        r -= r.mean()
        t -= t.mean()

        mse = np.mean(
            (r - t) ** 2
        )

        nmse = (
            mse
            / np.mean(t**2)
        )

        pcc = np.corrcoef(
            r,
            t,
        )[0, 1]

        std_ratio = (
            np.std(r)
            / np.std(t)
        )

        out.append(
            (
                float(mse),
                float(nmse),
                float(pcc),
                float(std_ratio),
            )
        )

    return out


# ============================================================
# FOURIER STARLET
# ============================================================
class FourierStarlet:
    """
    Exact periodic Fourier implementation of the
    B3-spline a-trous Starlet.

    Only J detail bands are penalised.
    """

    def __init__(
        self,
        H,
        W,
        J,
        device,
    ):
        wy = (
            2.0
            * np.pi
            * torch.fft.fftfreq(
                H,
                device=device,
            )
        )

        wx = (
            2.0
            * np.pi
            * torch.fft.fftfreq(
                W,
                device=device,
            )
        )

        WY, WX = torch.meshgrid(
            wy,
            wx,
            indexing="ij",
        )

        cumulative = torch.ones(
            (H, W),
            dtype=torch.float32,
            device=device,
        )

        bands = []

        for j in range(J):
            step = 2**j

            hy = (
                6.0
                + 8.0
                * torch.cos(
                    step * WY
                )
                + 2.0
                * torch.cos(
                    2.0 * step * WY
                )
            ) / 16.0

            hx = (
                6.0
                + 8.0
                * torch.cos(
                    step * WX
                )
                + 2.0
                * torch.cos(
                    2.0 * step * WX
                )
            ) / 16.0

            h2 = hy * hx

            bands.append(
                cumulative
                * (1.0 - h2)
            )

            cumulative = (
                cumulative
                * h2
            )

        # Shape:
        # (1,J,H,W)
        self.B = torch.stack(
            bands,
            dim=0,
        ).unsqueeze(0)

        self.norm2 = float(
            torch.max(
                torch.sum(
                    self.B[0] ** 2,
                    dim=0,
                )
            ).item()
        )

    def analysis(self, x):
        # x: (B,H,W)

        X = torch.fft.fft2(
            x,
            norm="ortho",
        )

        return torch.fft.ifft2(
            self.B
            * X.unsqueeze(1),
            norm="ortho",
        ).real

    def adjoint(self, coeff):
        # coeff: (B,J,H,W)

        C = torch.fft.fft2(
            coeff,
            norm="ortho",
        )

        X = torch.sum(
            self.B * C,
            dim=1,
        )

        return torch.fft.ifft2(
            X,
            norm="ortho",
        ).real


# ============================================================
# MCALENS P(k) -> 2-D MAP
# ============================================================
def make_pk_map(
    pk,
    H,
    W,
):
    """
    Same radial indexing convention used in
    CosmoStat get_ima_spectrum_map().
    """
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

    ky = (
        np.arange(H)
        - H / 2.0
    )

    kx = (
        np.arange(W)
        - W / 2.0
    )

    KY, KX = np.meshgrid(
        ky,
        kx,
        indexing="ij",
    )

    ir = np.sqrt(
        KX**2
        + KY**2
    ).astype(
        np.int64
    )

    ir[
        ir > npk
    ] = npk

    centered = ext[ir]

    return np.fft.ifftshift(
        centered
    )


# ============================================================
# GAUSSIAN PRIOR PROX
# ============================================================
def prox_gaussian(
    v,
    pk,
    tau_beta,
):
    """
    Proximal map for

        beta/2 sum_k |xhat(k)|^2 / P(k)
    """
    V = torch.fft.fft2(
        v,
        norm="ortho",
    )

    factor = torch.where(
        pk > 0,
        pk
        / (
            pk
            + tau_beta
        ),
        torch.zeros_like(
            pk
        ),
    )

    return torch.fft.ifft2(
        factor * V,
        norm="ortho",
    ).real


# ============================================================
# PDHG SOLVER
# ============================================================
@torch.no_grad()
def solve(
    op,
    star,
    y,
    mask,
    pk,
    beta,
    lam,
    alpha,
    x0,
):
    """
    Solve

      min_{xG,xS}

       1/2 || M[A(xG+xS)-y] ||_2^2

       + beta/2 sum_k |F xG(k)|^2 / P(k)

       + lambda sum_j alpha_j ||K_j xS||_1
    """

    xg = x0.clone()

    xs = torch.zeros_like(
        xg
    )

    xg_bar = xg.clone()
    xs_bar = xs.clone()

    q = torch.zeros_like(
        y,
        dtype=torch.complex64,
    )

    p = torch.zeros(
        (
            y.shape[0],
            J,
            N,
            N,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    limit = (
        lam
        * alpha
    )

    last_rel = None

    for it in range(
        1,
        NITER + 1,
    ):
        # ----------------------------------------------------
        # DATA DUAL
        # ----------------------------------------------------
        pred = op_forward_batch(
            op,
            xg_bar + xs_bar,
        )

        u = (
            q
            + SIGMA
            * mask
            * pred
        )

        q = (
            u
            - SIGMA * y
        ) / (
            1.0
            + SIGMA
        )

        # ----------------------------------------------------
        # STARLET DUAL
        # ----------------------------------------------------
        p = (
            p
            + SIGMA
            * star.analysis(
                xs_bar
            )
        )

        p = torch.maximum(
            torch.minimum(
                p,
                limit,
            ),
            -limit,
        )

        # ----------------------------------------------------
        # COMMON DATA ADJOINT
        # ----------------------------------------------------
        grad = op_adjoint_batch(
            op,
            mask * q,
        )

        if torch.is_complex(
            grad
        ):
            grad = grad.real

        oldg = xg
        olds = xs

        # ----------------------------------------------------
        # GAUSSIAN COMPONENT
        # ----------------------------------------------------
        xg = prox_gaussian(
            xg
            - TAU * grad,
            pk,
            TAU * beta,
        )

        # ----------------------------------------------------
        # SPARSE COMPONENT
        # ----------------------------------------------------
        xs = (
            xs
            - TAU
            * (
                grad
                + star.adjoint(p)
            )
        )

        # ----------------------------------------------------
        # EXTRAPOLATION
        # ----------------------------------------------------
        xg_bar = (
            xg
            + THETA
            * (
                xg
                - oldg
            )
        )

        xs_bar = (
            xs
            + THETA
            * (
                xs
                - olds
            )
        )

        # ----------------------------------------------------
        # FINAL RELATIVE CHANGE
        # ----------------------------------------------------
        if it == NITER:
            dx = torch.sqrt(
                torch.sum(
                    (
                        xg
                        - oldg
                    ) ** 2,
                    dim=(-2, -1),
                )
                +
                torch.sum(
                    (
                        xs
                        - olds
                    ) ** 2,
                    dim=(-2, -1),
                )
            )

            xn = torch.sqrt(
                torch.sum(
                    xg**2,
                    dim=(-2, -1),
                )
                +
                torch.sum(
                    xs**2,
                    dim=(-2, -1),
                )
            )

            last_rel = (
                dx
                / torch.clamp(
                    xn,
                    min=1e-12,
                )
            )

    return (
        xg + xs,
        xg,
        xs,
        last_rel,
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================
mask_full = np.load(
    MASK_FILE
).astype(bool)

truths = []
gammas = []
masks = []
meta = []

for c in COSMOS:
    for r in REALS:
        fn = (
            BASE
            / f"C{c}_R{r}.npz"
        )

        with np.load(fn) as z:
            truth_full = np.asarray(
                z["truth"],
                dtype=np.float32,
            )

            gamma_full = np.asarray(
                z["gamma"],
                dtype=np.complex64,
            )

        for pidx, (
            r0,
            r1,
        ) in enumerate(
            PATCH_ROWS
        ):
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

            truths.append(
                truth
            )

            gammas.append(
                gamma * mask
            )

            masks.append(
                mask
            )

            meta.append(
                (
                    c,
                    r,
                    pidx,
                )
            )

truths = np.asarray(
    truths,
    dtype=np.float32,
)

gammas = np.asarray(
    gammas,
    dtype=np.complex64,
)

masks = np.asarray(
    masks,
    dtype=np.float32,
)

assert truths.shape == (
    240,
    N,
    N,
)

assert gammas.shape == (
    240,
    N,
    N,
)

assert masks.shape == (
    240,
    N,
    N,
)


# ============================================================
# OPERATORS
# ============================================================
op = WeakLensingOperator(
    N,
    N,
    DEVICE,
)

ks = KaiserSquires(
    op
)

star = FourierStarlet(
    N,
    N,
    J,
    DEVICE,
)

pk_1d = np.load(
    PK_FILE
).astype(
    np.float64
)

pk_np = make_pk_map(
    pk_1d,
    N,
    N,
)

pk = torch.as_tensor(
    pk_np,
    dtype=torch.float32,
    device=DEVICE,
).unsqueeze(0)

alpha = torch.as_tensor(
    ALPHA,
    dtype=torch.float32,
    device=DEVICE,
).view(
    1,
    J,
    1,
    1,
)


# ============================================================
# INITIAL RESULT TABLE
# ============================================================
rows = []

# ------------------------------------------------------------
# First use any partially completed JOINT run.
# ------------------------------------------------------------
if CSV.exists():
    old_joint = pd.read_csv(
        CSV
    )

    rows.extend(
        old_joint.to_dict(
            "records"
        )
    )

    print(
        "Loaded joint-grid rows:",
        len(old_joint),
    )

# ------------------------------------------------------------
# Import existing lambda=0.05 grid.
# ------------------------------------------------------------
if OLD_CSV.exists():
    old = pd.read_csv(
        OLD_CSV
    )

    needed = {
        "lambda",
        "beta",
        "c",
        "r",
        "p",
        "mse",
        "nmse",
        "pcc",
        "std_ratio",
    }

    missing = (
        needed
        - set(old.columns)
    )

    if missing:
        raise RuntimeError(
            f"{OLD_CSV} is missing columns: "
            f"{sorted(missing)}"
        )

    for _, row in old.iterrows():
        rows.append(
            row.to_dict()
        )

    print(
        "Imported old lambda=0.05 rows:",
        len(old),
    )


# ============================================================
# CLEAN / DEDUPLICATE
# ============================================================
if rows:
    existing = pd.DataFrame(
        rows
    )

    existing = (
        existing
        .drop_duplicates(
            subset=[
                "lambda",
                "beta",
                "c",
                "r",
                "p",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    rows = existing.to_dict(
        "records"
    )

else:
    existing = pd.DataFrame()


def combo_complete(
    table,
    lam,
    beta,
):
    if len(table) == 0:
        return False

    q = table[
        np.isclose(
            table["lambda"],
            lam,
            rtol=0,
            atol=1e-12,
        )
        &
        np.isclose(
            table["beta"],
            beta,
            rtol=0,
            atol=1e-12,
        )
    ]

    return len(q) == 240


# ============================================================
# HEADER
# ============================================================
print()
print("=" * 105)
print("GAUSSIAN + STARLET — JOINT VALIDATION GRID")
print("=" * 105)

print(
    "device          :",
    DEVICE,
)

print(
    "cases           :",
    len(truths),
)

print(
    "lambda grid     :",
    LAMBDA_GRID,
)

print(
    "beta grid       :",
    BETA_GRID,
)

print(
    "iterations      :",
    NITER,
)

print(
    "batch           :",
    BATCH,
)

print(
    "Starlet norm^2  :",
    star.norm2,
)

print(
    "Pk file         :",
    PK_FILE,
)

print()


# ============================================================
# RUN MISSING COMBINATIONS
# ============================================================
for lam in LAMBDA_GRID:
    for beta in BETA_GRID:

        current = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame()
        )

        if combo_complete(
            current,
            lam,
            beta,
        ):
            q = current[
                np.isclose(
                    current["lambda"],
                    lam,
                    atol=1e-12,
                    rtol=0,
                )
                &
                np.isclose(
                    current["beta"],
                    beta,
                    atol=1e-12,
                    rtol=0,
                )
            ]

            print(
                f"SKIP lambda={lam:.3f}, "
                f"beta={beta:.6e}: "
                f"already complete "
                f"(NMSE={q.nmse.mean():.9f})"
            )

            continue

        print()
        print("=" * 105)
        print(
            f"LAMBDA = {lam:.5f}   "
            f"BETA = {beta:.6e}"
        )
        print("=" * 105)

        # Remove any incomplete copy of this combination
        # before recomputing it from scratch.
        if rows:
            current = pd.DataFrame(
                rows
            )

            keep = ~(
                np.isclose(
                    current["lambda"],
                    lam,
                    atol=1e-12,
                    rtol=0,
                )
                &
                np.isclose(
                    current["beta"],
                    beta,
                    atol=1e-12,
                    rtol=0,
                )
            )

            current = current[
                keep
            ].copy()

            rows = current.to_dict(
                "records"
            )

        combo_rows = []

        combo_t0 = (
            time.perf_counter()
        )

        for ib in range(
            0,
            len(truths),
            BATCH,
        ):
            ie = min(
                ib + BATCH,
                len(truths),
            )

            y = torch.as_tensor(
                gammas[ib:ie],
                dtype=torch.complex64,
                device=DEVICE,
            )

            m = torch.as_tensor(
                masks[ib:ie],
                dtype=torch.float32,
                device=DEVICE,
            )

            # Same initialization used in the previous run.
            x0 = ks_reconstruct_batch(
                ks,
                y,
                smooth_sigma=1.0,
            )

            if torch.is_complex(
                x0
            ):
                x0 = x0.real

            x0 = x0.to(
                dtype=torch.float32
            )

            batch_t0 = (
                time.perf_counter()
            )

            (
                rec,
                xg,
                xs,
                rel,
            ) = solve(
                op=op,
                star=star,
                y=y,
                mask=m,
                pk=pk,
                beta=beta,
                lam=lam,
                alpha=alpha,
                x0=x0,
            )

            batch_runtime = (
                time.perf_counter()
                - batch_t0
            )

            rec_np = (
                rec
                .cpu()
                .numpy()
            )

            xg_np = (
                xg
                .cpu()
                .numpy()
            )

            xs_np = (
                xs
                .cpu()
                .numpy()
            )

            rel_np = (
                rel
                .cpu()
                .numpy()
            )

            mets = ms_metrics_batch(
                rec_np,
                truths[ib:ie],
                masks[ib:ie],
            )

            for j, met in enumerate(
                mets
            ):
                idx = ib + j

                c, r, pidx = (
                    meta[idx]
                )

                mse, nmse, pcc, sr = (
                    met
                )

                obs_mask = masks[
                    idx
                ].astype(bool)

                combo_rows.append(
                    {
                        "lambda": lam,
                        "beta": beta,

                        "c": c,
                        "r": r,
                        "p": pidx,

                        "mse": mse,
                        "nmse": nmse,
                        "pcc": pcc,
                        "std_ratio": sr,

                        "std_gaussian": float(
                            np.std(
                                xg_np[j][
                                    obs_mask
                                ]
                            )
                        ),

                        "std_sparse": float(
                            np.std(
                                xs_np[j][
                                    obs_mask
                                ]
                            )
                        ),

                        "last_rel_change": float(
                            rel_np[j]
                        ),

                        "runtime_batch": (
                            batch_runtime
                        ),

                        "batch_size": (
                            ie - ib
                        ),
                    }
                )

            print(
                f"{ie:3d}/240  "
                f"runtime="
                f"{batch_runtime:.3f}s  "
                f"mean_rel="
                f"{np.mean(rel_np):.3e}",
                flush=True,
            )

        elapsed = (
            time.perf_counter()
            - combo_t0
        )

        rows.extend(
            combo_rows
        )

        # --------------------------------------------
        # SAVE AFTER EVERY COMPLETE COMBINATION
        # --------------------------------------------
        save = pd.DataFrame(
            rows
        )

        save = (
            save
            .drop_duplicates(
                subset=[
                    "lambda",
                    "beta",
                    "c",
                    "r",
                    "p",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "lambda",
                    "beta",
                    "c",
                    "r",
                    "p",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        save.to_csv(
            CSV,
            index=False,
        )

        rows = save.to_dict(
            "records"
        )

        q = save[
            np.isclose(
                save["lambda"],
                lam,
                atol=1e-12,
                rtol=0,
            )
            &
            np.isclose(
                save["beta"],
                beta,
                atol=1e-12,
                rtol=0,
            )
        ]

        print()
        print(
            f"lambda={lam:.5f}, "
            f"beta={beta:.6e}: "
            f"NMSE={q.nmse.mean():.9f}  "
            f"PCC={q.pcc.mean():.6f}  "
            f"std={q.std_ratio.mean():.6f}  "
            f"elapsed={elapsed:.1f}s"
        )


# ============================================================
# FINAL TABLE
# ============================================================
tab = pd.DataFrame(
    rows
)

tab = (
    tab
    .drop_duplicates(
        subset=[
            "lambda",
            "beta",
            "c",
            "r",
            "p",
        ],
        keep="last",
    )
    .sort_values(
        [
            "lambda",
            "beta",
            "c",
            "r",
            "p",
        ]
    )
    .reset_index(
        drop=True
    )
)

tab.to_csv(
    CSV,
    index=False,
)


# ============================================================
# VERIFY GRID COMPLETENESS
# ============================================================
print()
print("=" * 105)
print("GRID COMPLETENESS")
print("=" * 105)

for lam in LAMBDA_GRID:
    for beta in BETA_GRID:
        q = tab[
            np.isclose(
                tab["lambda"],
                lam,
                atol=1e-12,
                rtol=0,
            )
            &
            np.isclose(
                tab["beta"],
                beta,
                atol=1e-12,
                rtol=0,
            )
        ]

        print(
            f"lambda={lam:.3f} "
            f"beta={beta:.4g} "
            f"n={len(q)}"
        )

        if len(q) != 240:
            raise RuntimeError(
                "Incomplete validation grid."
            )


# ============================================================
# GLOBAL SUMMARY
# ============================================================
summary = (
    tab
    .groupby(
        [
            "lambda",
            "beta",
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

        gaussian_std_mean=(
            "std_gaussian",
            "mean",
        ),

        sparse_std_mean=(
            "std_sparse",
            "mean",
        ),

        rel_change_mean=(
            "last_rel_change",
            "mean",
        ),

        rel_change_max=(
            "last_rel_change",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        "nmse_mean"
    )
)

summary.to_csv(
    SUMMARY_CSV,
    index=False,
)

print()
print("=" * 125)
print("GLOBAL JOINT VALIDATION SUMMARY")
print("=" * 125)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# ============================================================
# BY COSMOLOGY
# ============================================================
byc = (
    tab
    .groupby(
        [
            "lambda",
            "beta",
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
    .sort_values(
        [
            "lambda",
            "beta",
            "c",
        ]
    )
)

byc.to_csv(
    BYCOSMO_CSV,
    index=False,
)

print()
print("=" * 105)
print("BY COSMOLOGY")
print("=" * 105)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# ============================================================
# BEST VALIDATION POINT
# ============================================================
best = summary.iloc[0]

best_lambda = float(
    best["lambda"]
)

best_beta = float(
    best["beta"]
)

print()
print("=" * 105)
print("BEST VALIDATION POINT")
print("=" * 105)

print(
    "lambda         :",
    best_lambda,
)

print(
    "beta           :",
    best_beta,
)

print(
    "mean NMSE      :",
    float(
        best["nmse_mean"]
    ),
)

print(
    "mean PCC       :",
    float(
        best["pcc_mean"]
    ),
)

print(
    "mean std ratio :",
    float(
        best["std_ratio_mean"]
    ),
)


# ============================================================
# BRACKETING DIAGNOSTIC
# ============================================================
print()
print("=" * 105)
print("BRACKETING DIAGNOSTIC")
print("=" * 105)

lambda_edge = (
    np.isclose(
        best_lambda,
        min(LAMBDA_GRID),
    )
    or
    np.isclose(
        best_lambda,
        max(LAMBDA_GRID),
    )
)

beta_edge = (
    np.isclose(
        best_beta,
        min(BETA_GRID),
    )
    or
    np.isclose(
        best_beta,
        max(BETA_GRID),
    )
)

print(
    "lambda optimum on edge :",
    lambda_edge,
)

print(
    "beta optimum on edge   :",
    beta_edge,
)

if not lambda_edge:
    print(
        "lambda minimum is bracketed."
    )
else:
    print(
        "lambda minimum is on the grid edge;"
        " extend ONLY that side before freezing."
    )

if not beta_edge:
    print(
        "beta minimum is bracketed."
    )
else:
    print(
        "beta minimum is on the grid edge;"
        " extend ONLY that side before freezing."
    )


# ============================================================
# REFERENCE METHODS
# ============================================================
print()
print("=" * 105)
print("REFERENCE — SAME SQUARE VALIDATION BENCHMARK")
print("=" * 105)

references = {
    "KS": 0.7288416041,
    "GLIMPSE": 0.828271,
    "STARLET_NOMASK": 0.695238,
    "STARLET_MASK": 0.682278,
    "MCALENS": 0.658007,
}

best_nmse = float(
    best["nmse_mean"]
)

for name, nmse in references.items():
    gain = (
        100.0
        * (
            nmse
            - best_nmse
        )
        / nmse
    )

    print(
        f"{name:18s} "
        f"NMSE={nmse:.9f}   "
        f"best G+S gain={gain:+.3f}%"
    )


print()
print("=" * 105)
print("FILES")
print("=" * 105)

print(CSV)
print(SUMMARY_CSV)
print(BYCOSMO_CSV)

print()
print("JOINT VALIDATION COMPLETE")
