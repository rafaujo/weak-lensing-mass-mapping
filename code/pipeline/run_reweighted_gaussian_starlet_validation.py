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
# CONFIG
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
    str(PATHS.work_root / 'validation/reweighted_gaussian_starlet')
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CSV = OUT / "reweighted_gaussian_starlet_validation.csv"
SUMMARY_CSV = OUT / "reweighted_gaussian_starlet_summary.csv"
BYCOSMO_CSV = OUT / "reweighted_gaussian_starlet_by_cosmology.csv"
WEIGHT_CSV = OUT / "reweighted_gaussian_starlet_weight_stats.csv"

# ------------------------------------------------------------
# Frozen first-stage parameters
# ------------------------------------------------------------
N = 176
J = 4

LAMBDA = 0.05
BETA = 0.001

ALPHA = np.array(
    [
        1.0,
        0.225148,
        0.096029,
        0.046286,
    ],
    dtype=np.float32,
)

TAU = 0.50
SIGMA = 0.50
THETA = 1.0

NITER_STAGE1 = 6000
NITER_STAGE2 = 6000

BATCH = 16

# ------------------------------------------------------------
# Reweighting
#
# q small  -> stronger debiasing
# q large  -> more conservative
# ------------------------------------------------------------
Q_GRID = [
    1.0,
    2.0,
    3.0,
]

WEIGHT_POWER = 2.0

# Never remove the sparse penalty completely.
ETA_MIN = 0.05

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

MAP_SHAPE = (
    3,
    10,
    8,
    N,
    N,
)

# ------------------------------------------------------------
# Stage-1 cache
# ------------------------------------------------------------
STAGE1_REC = OUT / "stage1_gaussian_starlet_rec.npy"
STAGE1_G = OUT / "stage1_gaussian_component.npy"
STAGE1_S = OUT / "stage1_sparse_component.npy"

# Expected baseline from previous validation.
EXPECTED_BASELINE_NMSE = 0.6469240253547854
EXPECTED_BASELINE_PCC = 0.5927316056309856


# ============================================================
# BATCH ADAPTERS
# ============================================================
def op_forward_batch(op, x):
    if x.ndim == 2:
        return op.forward(x)

    return torch.vmap(
        op.forward
    )(x)


def op_adjoint_batch(op, y):
    if y.ndim == 2:
        return op.adjoint(y)

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

    return torch.vmap(
        lambda yy: ks.reconstruct(
            yy,
            smooth_sigma=smooth_sigma,
        )
    )(y)


# ============================================================
# FOURIER STARLET
# ============================================================
class FourierStarlet:
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
                + 8.0 * torch.cos(step * WY)
                + 2.0 * torch.cos(2.0 * step * WY)
            ) / 16.0

            hx = (
                6.0
                + 8.0 * torch.cos(step * WX)
                + 2.0 * torch.cos(2.0 * step * WX)
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
# P(k)
# ============================================================
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

    centered = ext[
        ir
    ]

    return np.fft.ifftshift(
        centered
    )


# ============================================================
# GAUSSIAN PROX
# ============================================================
def prox_gaussian(
    v,
    pk,
    tau_beta,
):
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
# METRICS
# ============================================================
def ms_metrics_batch(
    rec,
    truth,
    masks,
):
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
# GENERIC GAUSSIAN + WEIGHTED-STARLET SOLVER
# ============================================================
@torch.no_grad()
def solve_weighted(
    op,
    star,
    y,
    mask,
    pk,
    alpha,
    xg0,
    xs0,
    weights,
    niter,
):
    """
    Solve, for FIXED weights:

      1/2 || M[A(xG+xS)-y] ||^2

      + beta/2 sum |FxG|^2 / Pk

      + lambda sum_j,p alpha_j w_jp |(KxS)_jp|

    Once w is fixed, this remains a convex problem.
    """

    xg = xg0.clone()
    xs = xs0.clone()

    xg_bar = xg.clone()
    xs_bar = xs.clone()

    qdual = torch.zeros_like(
        y,
        dtype=torch.complex64,
    )

    pdual = torch.zeros(
        (
            y.shape[0],
            J,
            N,
            N,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    # Weighted l_infinity dual ball.
    limit = (
        LAMBDA
        * alpha
        * weights
    )

    last_rel = None

    for it in range(
        1,
        niter + 1,
    ):
        # ----------------------------------------------------
        # DATA DUAL
        # ----------------------------------------------------
        pred = op_forward_batch(
            op,
            xg_bar + xs_bar,
        )

        u = (
            qdual
            + SIGMA
            * mask
            * pred
        )

        qdual = (
            u
            - SIGMA * y
        ) / (
            1.0
            + SIGMA
        )

        # ----------------------------------------------------
        # WEIGHTED STARLET DUAL
        # ----------------------------------------------------
        pdual = (
            pdual
            + SIGMA
            * star.analysis(
                xs_bar
            )
        )

        pdual = torch.maximum(
            torch.minimum(
                pdual,
                limit,
            ),
            -limit,
        )

        # ----------------------------------------------------
        # COMMON DATA GRADIENT
        # ----------------------------------------------------
        grad = op_adjoint_batch(
            op,
            mask * qdual,
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
            TAU * BETA,
        )

        # ----------------------------------------------------
        # SPARSE COMPONENT
        # ----------------------------------------------------
        xs = (
            xs
            - TAU
            * (
                grad
                + star.adjoint(
                    pdual
                )
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

        if it == niter:
            dx = torch.sqrt(
                torch.sum(
                    (
                        xg - oldg
                    ) ** 2,
                    dim=(-2, -1),
                )
                +
                torch.sum(
                    (
                        xs - olds
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
# REWEIGHTING RULE
# ============================================================
@torch.no_grad()
def make_reweighting(
    star,
    xs0,
    alpha,
    q,
):
    """
    Smooth support debiasing.

    c0 = K xs_stage1

    threshold_j = q * lambda * alpha_j

    w = eta +
        (1-eta) /
        [1 + (|c0| / threshold_j)^p]

    Small coefficients:
        w ~ 1
        -> retain original regularisation.

    Large coefficients:
        w -> eta
        -> much less shrinkage.
    """

    c0 = star.analysis(
        xs0
    )

    threshold = (
        q
        * LAMBDA
        * alpha
    )

    ratio = (
        torch.abs(c0)
        / torch.clamp(
            threshold,
            min=1e-12,
        )
    )

    weights = (
        ETA_MIN
        +
        (
            1.0
            - ETA_MIN
        )
        / (
            1.0
            + ratio**WEIGHT_POWER
        )
    )

    return (
        weights,
        c0,
    )


# ============================================================
# LOAD VALIDATION
# ============================================================
print()
print("=" * 110)
print("REWEIGHTED GAUSSIAN + STARLET — VALIDATION")
print("=" * 110)

mask_full = np.load(
    MASK_FILE
).astype(bool)

truths = []
gammas = []
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

            gamma_full = np.asarray(
                z["gamma"],
                dtype=np.complex64,
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

            mask = mask_full[
                r0:r1,
                :
            ].copy()

            masks.append(
                mask
            )

            gammas.append(
                gamma_full[
                    r0:r1,
                    :
                ].copy()
                * mask
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

pk1d = np.load(
    PK_FILE
).astype(
    np.float64
)

pk2d = make_pk_map(
    pk1d,
    N,
    N,
)

pk = torch.as_tensor(
    pk2d,
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

ones_weight = torch.ones(
    (
        1,
        J,
        N,
        N,
    ),
    dtype=torch.float32,
    device=DEVICE,
)

print(
    "device          :",
    DEVICE,
)

print(
    "cases           :",
    240,
)

print(
    "lambda          :",
    LAMBDA,
)

print(
    "beta            :",
    BETA,
)

print(
    "q grid          :",
    Q_GRID,
)

print(
    "eta min         :",
    ETA_MIN,
)

print(
    "weight power    :",
    WEIGHT_POWER,
)

print(
    "stage1 iter     :",
    NITER_STAGE1,
)

print(
    "stage2 iter     :",
    NITER_STAGE2,
)

print(
    "Starlet norm²   :",
    star.norm2,
)


# ============================================================
# CACHE HELPERS
# ============================================================
def create_memmap(
    path,
):
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=MAP_SHAPE,
    )


def flat_to_grid(
    arr_flat,
    grid,
    ib,
    ie,
):
    for j in range(
        ie - ib
    ):
        idx = ib + j

        c, r, p = meta[
            idx
        ]

        grid[
            c,
            r,
            p,
        ] = arr_flat[j]


# ============================================================
# STAGE 1
# ============================================================
if (
    STAGE1_REC.exists()
    and STAGE1_G.exists()
    and STAGE1_S.exists()
):
    print()
    print(
        "Loading cached Stage-1 maps..."
    )

    stage1_rec_grid = np.load(
        STAGE1_REC,
        mmap_mode="r",
    )

    stage1_g_grid = np.load(
        STAGE1_G,
        mmap_mode="r",
    )

    stage1_s_grid = np.load(
        STAGE1_S,
        mmap_mode="r",
    )

else:
    print()
    print("=" * 110)
    print("STAGE 1 — FROZEN GAUSSIAN + STARLET")
    print("=" * 110)

    stage1_rec_grid = create_memmap(
        STAGE1_REC
    )

    stage1_g_grid = create_memmap(
        STAGE1_G
    )

    stage1_s_grid = create_memmap(
        STAGE1_S
    )

    for ib in range(
        0,
        240,
        BATCH,
    ):
        ie = min(
            ib + BATCH,
            240,
        )

        y = torch.as_tensor(
            gammas[
                ib:ie
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        m = torch.as_tensor(
            masks[
                ib:ie
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        x0 = ks_reconstruct_batch(
            ks,
            y,
            smooth_sigma=1.0,
        )

        if torch.is_complex(
            x0
        ):
            x0 = x0.real

        x0 = x0.float()

        xs0 = torch.zeros_like(
            x0
        )

        t0 = time.perf_counter()

        (
            rec,
            xg,
            xs,
            rel,
        ) = solve_weighted(
            op=op,
            star=star,
            y=y,
            mask=m,
            pk=pk,
            alpha=alpha,
            xg0=x0,
            xs0=xs0,
            weights=ones_weight,
            niter=NITER_STAGE1,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        rec_np = (
            rec.cpu().numpy()
        )

        xg_np = (
            xg.cpu().numpy()
        )

        xs_np = (
            xs.cpu().numpy()
        )

        flat_to_grid(
            rec_np,
            stage1_rec_grid,
            ib,
            ie,
        )

        flat_to_grid(
            xg_np,
            stage1_g_grid,
            ib,
            ie,
        )

        flat_to_grid(
            xs_np,
            stage1_s_grid,
            ib,
            ie,
        )

        stage1_rec_grid.flush()
        stage1_g_grid.flush()
        stage1_s_grid.flush()

        print(
            f"{ie:3d}/240  "
            f"time={elapsed:.2f}s  "
            f"rel={rel.mean().item():.3e}",
            flush=True,
        )


# ============================================================
# FLATTEN STAGE 1
# ============================================================
stage1_rec = np.empty_like(
    truths
)

stage1_g = np.empty_like(
    truths
)

stage1_s = np.empty_like(
    truths
)

for i, (
    c,
    r,
    p,
) in enumerate(meta):

    stage1_rec[i] = (
        stage1_rec_grid[
            c,
            r,
            p,
        ]
    )

    stage1_g[i] = (
        stage1_g_grid[
            c,
            r,
            p,
        ]
    )

    stage1_s[i] = (
        stage1_s_grid[
            c,
            r,
            p,
        ]
    )


# ============================================================
# VERIFY STAGE 1
# ============================================================
base_metrics = ms_metrics_batch(
    stage1_rec,
    truths,
    masks,
)

base_metrics = np.asarray(
    base_metrics
)

base_nmse = float(
    base_metrics[:, 1].mean()
)

base_pcc = float(
    base_metrics[:, 2].mean()
)

base_std = float(
    base_metrics[:, 3].mean()
)

print()
print("=" * 110)
print("STAGE-1 SANITY")
print("=" * 110)

print(
    f"NMSE      : {base_nmse:.9f}"
)

print(
    f"PCC       : {base_pcc:.9f}"
)

print(
    f"std ratio : {base_std:.9f}"
)

print(
    f"expected NMSE : "
    f"{EXPECTED_BASELINE_NMSE:.9f}"
)

if abs(
    base_nmse
    - EXPECTED_BASELINE_NMSE
) > 2e-3:

    raise RuntimeError(
        "Stage-1 does not reproduce the frozen "
        "Gaussian+Starlet validation benchmark."
    )


# ============================================================
# RESULT TABLE
# ============================================================
rows = []

# Baseline stage 1.
for i, (
    c,
    r,
    p,
) in enumerate(meta):

    mse, nmse, pcc, sr = (
        base_metrics[i]
    )

    rows.append(
        {
            "method": "STAGE1",
            "q": np.nan,
            "c": c,
            "r": r,
            "p": p,
            "mse": mse,
            "nmse": nmse,
            "pcc": pcc,
            "std_ratio": sr,
            "std_gaussian": float(
                np.std(
                    stage1_g[i][
                        masks[i].astype(bool)
                    ]
                )
            ),
            "std_sparse": float(
                np.std(
                    stage1_s[i][
                        masks[i].astype(bool)
                    ]
                )
            ),
            "rel_change": np.nan,
        }
    )


# ============================================================
# STAGE 2 — REWEIGHTED SOLVES
# ============================================================
weight_rows = []

for qval in Q_GRID:

    print()
    print("=" * 110)
    print(
        f"STAGE 2 — REWEIGHTED q={qval:g}"
    )
    print("=" * 110)

    q_rec_file = (
        OUT
        / f"reweighted_q{qval:g}_rec.npy"
    )

    q_g_file = (
        OUT
        / f"reweighted_q{qval:g}_gaussian.npy"
    )

    q_s_file = (
        OUT
        / f"reweighted_q{qval:g}_sparse.npy"
    )

    q_rec_grid = create_memmap(
        q_rec_file
    )

    q_g_grid = create_memmap(
        q_g_file
    )

    q_s_grid = create_memmap(
        q_s_file
    )

    q_metric_rows = []

    for ib in range(
        0,
        240,
        BATCH,
    ):
        ie = min(
            ib + BATCH,
            240,
        )

        y = torch.as_tensor(
            gammas[
                ib:ie
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        m = torch.as_tensor(
            masks[
                ib:ie
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        xg0 = torch.as_tensor(
            stage1_g[
                ib:ie
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        xs0 = torch.as_tensor(
            stage1_s[
                ib:ie
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        weights, c0 = (
            make_reweighting(
                star=star,
                xs0=xs0,
                alpha=alpha,
                q=qval,
            )
        )

        # --------------------------------------------
        # Weight diagnostics
        # --------------------------------------------
        for j in range(
            J
        ):
            wj = (
                weights[
                    :,
                    j,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            cj = (
                torch.abs(
                    c0[
                        :,
                        j,
                    ]
                )
                .detach()
                .cpu()
                .numpy()
            )

            weight_rows.append(
                {
                    "q": qval,
                    "batch_start": ib,
                    "scale": j,
                    "weight_mean": float(
                        wj.mean()
                    ),
                    "weight_median": float(
                        np.median(wj)
                    ),
                    "frac_w_lt_050": float(
                        np.mean(
                            wj < 0.50
                        )
                    ),
                    "frac_w_lt_025": float(
                        np.mean(
                            wj < 0.25
                        )
                    ),
                    "frac_w_lt_010": float(
                        np.mean(
                            wj < 0.10
                        )
                    ),
                    "coeff_abs_mean": float(
                        cj.mean()
                    ),
                    "coeff_abs_median": float(
                        np.median(cj)
                    ),
                }
            )

        t0 = time.perf_counter()

        (
            rec,
            xg,
            xs,
            rel,
        ) = solve_weighted(
            op=op,
            star=star,
            y=y,
            mask=m,
            pk=pk,
            alpha=alpha,
            xg0=xg0,
            xs0=xs0,
            weights=weights,
            niter=NITER_STAGE2,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        rec_np = (
            rec.cpu().numpy()
        )

        xg_np = (
            xg.cpu().numpy()
        )

        xs_np = (
            xs.cpu().numpy()
        )

        rel_np = (
            rel.cpu().numpy()
        )

        mets = ms_metrics_batch(
            rec_np,
            truths[
                ib:ie
            ],
            masks[
                ib:ie
            ],
        )

        flat_to_grid(
            rec_np,
            q_rec_grid,
            ib,
            ie,
        )

        flat_to_grid(
            xg_np,
            q_g_grid,
            ib,
            ie,
        )

        flat_to_grid(
            xs_np,
            q_s_grid,
            ib,
            ie,
        )

        q_rec_grid.flush()
        q_g_grid.flush()
        q_s_grid.flush()

        for j, met in enumerate(
            mets
        ):
            idx = (
                ib + j
            )

            c, r, p = meta[
                idx
            ]

            mse, nmse, pcc, sr = (
                met
            )

            obs = masks[
                idx
            ].astype(bool)

            row = {
                "method": "REWEIGHTED",
                "q": qval,
                "c": c,
                "r": r,
                "p": p,
                "mse": mse,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": sr,

                "std_gaussian": float(
                    np.std(
                        xg_np[j][obs]
                    )
                ),

                "std_sparse": float(
                    np.std(
                        xs_np[j][obs]
                    )
                ),

                "rel_change": float(
                    rel_np[j]
                ),
            }

            q_metric_rows.append(
                row
            )

            rows.append(
                row
            )

        partial = pd.DataFrame(
            rows
        )

        partial.to_csv(
            CSV,
            index=False,
        )

        print(
            f"{ie:3d}/240  "
            f"time={elapsed:.2f}s  "
            f"rel={np.mean(rel_np):.3e}",
            flush=True,
        )

    qq = pd.DataFrame(
        q_metric_rows
    )

    print()
    print(
        f"q={qval:g}: "
        f"NMSE={qq.nmse.mean():.9f}  "
        f"PCC={qq.pcc.mean():.6f}  "
        f"std={qq.std_ratio.mean():.6f}"
    )


# ============================================================
# SAVE ALL RESULTS
# ============================================================
tab = pd.DataFrame(
    rows
)

tab.to_csv(
    CSV,
    index=False,
)

pd.DataFrame(
    weight_rows
).to_csv(
    WEIGHT_CSV,
    index=False,
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================
summary = (
    tab
    .groupby(
        [
            "method",
            "q",
        ],
        dropna=False,
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
            "rel_change",
            "mean",
        ),

        rel_change_max=(
            "rel_change",
            "max",
        ),
    )
    .reset_index()
)

# Add differences from Stage 1.
summary[
    "nmse_gain_vs_stage1_pct"
] = (
    100.0
    * (
        base_nmse
        - summary[
            "nmse_mean"
        ]
    )
    / base_nmse
)

summary[
    "pcc_delta_vs_stage1"
] = (
    summary[
        "pcc_mean"
    ]
    - base_pcc
)

summary[
    "std_delta_vs_stage1"
] = (
    summary[
        "std_ratio_mean"
    ]
    - base_std
)

summary = summary.sort_values(
    "nmse_mean"
)

summary.to_csv(
    SUMMARY_CSV,
    index=False,
)

print()
print("=" * 150)
print("GLOBAL VALIDATION SUMMARY")
print("=" * 150)

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
            "method",
            "q",
            "c",
        ],
        dropna=False,
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
    BYCOSMO_CSV,
    index=False,
)

print()
print("=" * 110)
print("BY COSMOLOGY")
print("=" * 110)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# ============================================================
# WEIGHT SUMMARY
# ============================================================
wtab = pd.DataFrame(
    weight_rows
)

wsummary = (
    wtab
    .groupby(
        [
            "q",
            "scale",
        ]
    )
    .agg(
        weight_mean=(
            "weight_mean",
            "mean",
        ),

        weight_median=(
            "weight_median",
            "mean",
        ),

        frac_w_lt_050=(
            "frac_w_lt_050",
            "mean",
        ),

        frac_w_lt_025=(
            "frac_w_lt_025",
            "mean",
        ),

        frac_w_lt_010=(
            "frac_w_lt_010",
            "mean",
        ),

        coeff_abs_mean=(
            "coeff_abs_mean",
            "mean",
        ),
    )
    .reset_index()
)

print()
print("=" * 120)
print("REWEIGHTING STATISTICS")
print("=" * 120)

print(
    wsummary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


# ============================================================
# DECISION TABLE
# ============================================================
print()
print("=" * 110)
print("REFERENCE")
print("=" * 110)

print(
    "Frozen G+S validation : "
    "NMSE=0.64692403  "
    "PCC=0.59273161  "
    "std=0.58883535"
)

print(
    "MCALens validation    : "
    "NMSE=0.65800700  "
    "PCC=0.59118900  "
    "std≈0.69117"
)

print()
print(
    "Desired direction:"
)

print(
    "  NMSE: stay near/below 0.6469"
)

print(
    "  PCC : stay near/above 0.5927"
)

print(
    "  std : rise above 0.5888 toward ~0.65-0.70"
)

print()
print(
    "DO NOT evaluate R10-R29 from this experiment yet."
)

print()
print("=" * 110)
print("FILES")
print("=" * 110)

print(CSV)
print(SUMMARY_CSV)
print(BYCOSMO_CSV)
print(WEIGHT_CSV)

print()
print("REWEIGHTED VALIDATION COMPLETE")
