from project_paths import PATHS
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(PATHS.code_root))

from weak_lensing_pipeline import WeakLensingOperator, KaiserSquires


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
    str(PATHS.work_root / 'validation/starlet_gaussian_prior')
)
OUT.mkdir(parents=True, exist_ok=True)

CSV = OUT / "gaussian_starlet_validation.csv"

N = 176
J = 4

LAMBDA = 0.05

# Only NEW hyperparameter is tuned here.
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
# METRICS
# ============================================================
def ms_metrics_batch(rec, truth, masks):
    out = []

    for i in range(rec.shape[0]):
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

        sr = (
            np.std(r)
            / np.std(t)
        )

        out.append(
            (
                float(mse),
                float(nmse),
                float(pcc),
                float(sr),
            )
        )

    return out


# ============================================================
# BATCH ADAPTERS FOR CANONICAL 2-D WEAK-LENSING OPERATOR
# ============================================================
def op_forward_batch(op, x):
    """
    Apply canonical WeakLensingOperator.forward independently
    over a leading batch dimension.

    x:
        (H,W)   -> (H,W)
        (B,H,W) -> (B,H,W), complex
    """
    if x.ndim == 2:
        return op.forward(x)

    if x.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), got {tuple(x.shape)}"
        )

    return torch.vmap(op.forward)(x)


def op_adjoint_batch(op, y):
    """
    Apply canonical WeakLensingOperator.adjoint independently
    over a leading batch dimension.

    y:
        (H,W)   -> (H,W)
        (B,H,W) -> (B,H,W)
    """
    if y.ndim == 2:
        return op.adjoint(y)

    if y.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), got {tuple(y.shape)}"
        )

    return torch.vmap(op.adjoint)(y)


def ks_reconstruct_batch(ks, y, smooth_sigma):
    """
    Batched adapter for the canonical KaiserSquires class.
    """
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
# FOURIER STARLET — BATCHED
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

    def adjoint(self, c):
        # c: (B,J,H,W)

        C = torch.fft.fft2(
            c,
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
# P(k) MAP — SAME CONVENTION AS MCALENS
# ============================================================
def make_pk_map(pk, H, W):
    pk = np.asarray(
        pk,
        dtype=np.float64,
    ).copy()

    pk[0] = 0.0

    ext = np.concatenate(
        [
            pk,
            np.zeros(1),
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
    ).astype(int)

    ir[ir > npk] = npk

    centered = ext[ir]

    return np.fft.ifftshift(
        centered
    )


# ============================================================
# GAUSSIAN PRIOR PROX
# ============================================================
def prox_gaussian(v, pk, tau_beta):
    V = torch.fft.fft2(
        v,
        norm="ortho",
    )

    factor = torch.where(
        pk > 0,
        pk / (
            pk + tau_beta
        ),
        torch.zeros_like(pk),
    )

    return torch.fft.ifft2(
        factor * V,
        norm="ortho",
    ).real


# ============================================================
# BATCH SOLVER
# ============================================================
@torch.no_grad()
def solve(
    op,
    star,
    y,
    mask,
    pk,
    beta,
    alpha,
    x0,
):
    xg = x0.clone()
    xs = torch.zeros_like(xg)

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

    lim = (
        LAMBDA
        * alpha
    )

    last_rel = np.nan

    for it in range(
        1,
        NITER + 1,
    ):

        # --------------------------------------------
        # data dual
        # --------------------------------------------
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
            1.0 + SIGMA
        )

        # --------------------------------------------
        # Starlet dual
        # --------------------------------------------
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
                lim,
            ),
            -lim,
        )

        # --------------------------------------------
        # common data gradient
        # --------------------------------------------
        grad = op_adjoint_batch(
            op,
            mask * q,
        )

        if torch.is_complex(grad):
            grad = grad.real

        oldg = xg
        olds = xs

        # --------------------------------------------
        # Gaussian component
        # --------------------------------------------
        xg = prox_gaussian(
            xg
            - TAU * grad,
            pk,
            TAU * beta,
        )

        # --------------------------------------------
        # sparse component
        # --------------------------------------------
        xs = (
            xs
            - TAU
            * (
                grad
                + star.adjoint(p)
            )
        )

        # --------------------------------------------
        # extrapolation
        # --------------------------------------------
        xg_bar = (
            xg
            + THETA
            * (xg - oldg)
        )

        xs_bar = (
            xs
            + THETA
            * (xs - olds)
        )

        if it == NITER:
            dx = torch.sqrt(
                torch.sum(
                    (xg-oldg)**2,
                    dim=(-2,-1),
                )
                +
                torch.sum(
                    (xs-olds)**2,
                    dim=(-2,-1),
                )
            )

            xn = torch.sqrt(
                torch.sum(
                    xg**2,
                    dim=(-2,-1),
                )
                +
                torch.sum(
                    xs**2,
                    dim=(-2,-1),
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
# LOAD ALL VALIDATION CASES
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

        for p, (r0, r1) in enumerate(
            PATCH_ROWS
        ):
            t = truth_full[
                r0:r1,
                :
            ].copy()

            g = gamma_full[
                r0:r1,
                :
            ].copy()

            m = mask_full[
                r0:r1,
                :
            ].copy()

            truths.append(t)

            # identical observation used by square benchmark
            gammas.append(
                g * m
            )

            masks.append(m)

            meta.append(
                (c, r, p)
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
    176,
    176,
)

print("=" * 90)
print("GAUSSIAN + STARLET — VALIDATION")
print("=" * 90)

print("device        :", DEVICE)
print("cases         :", len(truths))
print("lambda        :", LAMBDA)
print("beta grid     :", BETA_GRID)
print("iterations    :", NITER)
print("batch         :", BATCH)


# ============================================================
# OPERATORS
# ============================================================
op = WeakLensingOperator(
    N,
    N,
    DEVICE,
)

ks = KaiserSquires(op)

star = FourierStarlet(
    N,
    N,
    J,
    DEVICE,
)

print(
    "Starlet norm² :",
    star.norm2,
)

pk_1d = np.load(
    PK_FILE
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
)

# broadcast over batch
pk = pk.unsqueeze(0)

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
# RUN GRID
# ============================================================
rows = []

for beta in BETA_GRID:

    print()
    print("=" * 90)
    print(f"BETA = {beta:.6e}")
    print("=" * 90)

    t_beta = time.perf_counter()

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

        # Same mild KS initialization used previously.
        x0 = ks_reconstruct_batch(
            ks,
            y,
            smooth_sigma=1.0,
        )

        if torch.is_complex(x0):
            x0 = x0.real

        x0 = x0.to(
            dtype=torch.float32
        )

        t0 = time.perf_counter()

        rec, xg, xs, rel = solve(
            op=op,
            star=star,
            y=y,
            mask=m,
            pk=pk,
            beta=beta,
            alpha=alpha,
            x0=x0,
        )

        runtime = (
            time.perf_counter()
            - t0
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

        for j, met in enumerate(mets):

            idx = ib + j

            c, r, pidx = meta[idx]

            mse, nmse, pcc, sr = met

            rows.append(
                {
                    "beta": beta,
                    "lambda": LAMBDA,

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
                                masks[idx].astype(bool)
                            ]
                        )
                    ),

                    "std_sparse": float(
                        np.std(
                            xs_np[j][
                                masks[idx].astype(bool)
                            ]
                        )
                    ),

                    "last_rel_change": float(
                        rel_np[j]
                    ),

                    "runtime_batch": runtime,
                    "batch_size": ie-ib,
                }
            )

        print(
            f"{ie:3d}/240  "
            f"batch runtime={runtime:.3f}s  "
            f"mean rel={rel_np.mean():.3e}",
            flush=True,
        )

    elapsed = (
        time.perf_counter()
        - t_beta
    )

    tab_now = pd.DataFrame(
        rows
    )

    tab_now.to_csv(
        CSV,
        index=False,
    )

    q = tab_now[
        tab_now["beta"] == beta
    ]

    print()
    print(
        f"beta={beta:.6e}: "
        f"NMSE={q.nmse.mean():.9f}  "
        f"PCC={q.pcc.mean():.6f}  "
        f"std={q.std_ratio.mean():.6f}  "
        f"elapsed={elapsed:.1f}s"
    )


# ============================================================
# SUMMARY
# ============================================================
tab = pd.DataFrame(rows)

tab.to_csv(
    CSV,
    index=False,
)

summary = (
    tab.groupby("beta")
    .agg(
        n=("nmse", "size"),

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
    OUT
    / "gaussian_starlet_validation_summary.csv",
    index=False,
)

print()
print("=" * 110)
print("GLOBAL VALIDATION SUMMARY")
print("=" * 110)

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
    tab.groupby(
        ["beta", "c"]
    )
    .agg(
        n=("nmse", "size"),
        nmse_mean=("nmse", "mean"),
        pcc_mean=("pcc", "mean"),
        std_ratio_mean=(
            "std_ratio",
            "mean",
        ),
    )
    .reset_index()
)

print()
print("=" * 90)
print("BY COSMOLOGY")
print("=" * 90)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)

byc.to_csv(
    OUT
    / "gaussian_starlet_validation_by_cosmology.csv",
    index=False,
)


# ============================================================
# EXISTING REFERENCES
# ============================================================
print()
print("=" * 90)
print("REFERENCE — SAME SQUARE VALIDATION BENCHMARK")
print("=" * 90)

print("KS              : NMSE 0.728842")
print("GLIMPSE         : NMSE 0.828271")
print("STARLET_NOMASK  : NMSE 0.695238")
print("STARLET_MASK    : NMSE 0.682278")
print("MCALens Nsig=5  : NMSE 0.658007")

print()
print("Saved:", CSV)
print("VALIDATION COMPLETE")
