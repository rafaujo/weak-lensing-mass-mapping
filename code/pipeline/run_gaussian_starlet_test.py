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
# FROZEN CONFIGURATION
# ============================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE = Path(
    str(PATHS.work_root / 'test/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

# IMPORTANT:
# Estimated exclusively from VALIDATION truth maps R0-R9.
PK_FILE = Path(
    str(PATHS.prior_file)
)

OUT = Path(
    str(PATHS.work_root / 'test/starlet_gaussian_prior')
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CSV = OUT / "gaussian_starlet_test_metrics.csv"

SUMMARY_CSV = (
    OUT
    / "gaussian_starlet_test_summary.csv"
)

BYCOSMO_CSV = (
    OUT
    / "gaussian_starlet_test_by_cosmology.csv"
)

PAIRWISE_CSV = (
    OUT
    / "gaussian_starlet_test_pairwise_bootstrap.csv"
)

# ------------------------------------------------------------
# FROZEN HYPERPARAMETERS
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

NITER = 6000

TAU = 0.50
SIGMA = 0.50
THETA = 1.0

BATCH = 16

COSMOS = range(3)
REALS = range(10, 30)
PATCHES = range(8)

NBOOT = 30000
BOOT_SEED = 12345

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
# OUTPUT MAPS
# ============================================================
# Shape:
#   cosmology x test-realization x patch x y x x
#
MAP_SHAPE = (
    3,
    20,
    8,
    176,
    176,
)

REC_FILE = (
    OUT
    / "gaussian_starlet_test_reconstruction.npy"
)

GAUSS_FILE = (
    OUT
    / "gaussian_starlet_test_gaussian.npy"
)

SPARSE_FILE = (
    OUT
    / "gaussian_starlet_test_sparse.npy"
)


def open_output_map(path):
    """
    Resume-safe .npy memmap.
    """
    if path.exists():
        arr = np.load(
            path,
            mmap_mode="r+",
        )

        if arr.shape != MAP_SHAPE:
            raise RuntimeError(
                f"{path}: wrong shape "
                f"{arr.shape}, expected {MAP_SHAPE}"
            )

        return arr

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=MAP_SHAPE,
    )


rec_map = open_output_map(
    REC_FILE
)

gauss_map = open_output_map(
    GAUSS_FILE
)

sparse_map = open_output_map(
    SPARSE_FILE
)


# ============================================================
# BATCH ADAPTERS
# ============================================================
def op_forward_batch(op, x):
    if x.ndim == 2:
        return op.forward(x)

    if x.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), "
            f"got {tuple(x.shape)}"
        )

    return torch.vmap(
        op.forward
    )(x)


def op_adjoint_batch(op, y):
    if y.ndim == 2:
        return op.adjoint(y)

    if y.ndim != 3:
        raise ValueError(
            f"Expected (H,W) or (B,H,W), "
            f"got {tuple(y.shape)}"
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
            f"Expected (H,W) or (B,H,W), "
            f"got {tuple(y.shape)}"
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

        # Mass-sheet invariant.
        r -= r.mean()
        t -= t.mean()

        err = r - t

        mse = np.mean(
            err**2
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

        # (1,J,H,W)
        self.B = torch.stack(
            bands,
            dim=0,
        ).unsqueeze(0)

        self.norm2 = float(
            torch.max(
                torch.sum(
                    self.B[0]**2,
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
# P(k) MAP
# ============================================================
def make_pk_map(
    pk,
    H,
    W,
):
    """
    Same radial indexing convention used by
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
# FROZEN PDHG SOLVER
# ============================================================
@torch.no_grad()
def solve(
    op,
    star,
    y,
    mask,
    pk,
    alpha,
    x0,
):
    """
    FROZEN objective:

      min_{xG,xS}

        1/2 || M[A(xG+xS)-y] ||_2^2

        + BETA/2 sum_k |F xG(k)|^2 / P_kappa(k)

        + LAMBDA sum_j alpha_j ||K_j xS||_1
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
        LAMBDA
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
        # DATA ADJOINT
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
        # FINAL CONVERGENCE DIAGNOSTIC
        # ----------------------------------------------------
        if it == NITER:
            dx = torch.sqrt(
                torch.sum(
                    (
                        xg
                        - oldg
                    )**2,
                    dim=(-2, -1),
                )
                +
                torch.sum(
                    (
                        xs
                        - olds
                    )**2,
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
# LOAD TEST DATA
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

        if not fn.exists():
            raise FileNotFoundError(
                fn
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

            # Same observed square shear used by
            # KS / Starlet / GLIMPSE / MCALens.
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
    480,
    N,
    N,
)

assert gammas.shape == (
    480,
    N,
    N,
)

assert masks.shape == (
    480,
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
# RESUME SUPPORT
# ============================================================
if CSV.exists():
    old = pd.read_csv(
        CSV
    )

    old = (
        old
        .drop_duplicates(
            subset=[
                "c",
                "r",
                "p",
            ],
            keep="last",
        )
    )

    done = set(
        zip(
            old["c"].astype(int),
            old["r"].astype(int),
            old["p"].astype(int),
        )
    )

    rows = old.to_dict(
        "records"
    )

else:
    done = set()
    rows = []


# ============================================================
# HEADER
# ============================================================
print()
print("=" * 100)
print("GAUSSIAN + STARLET — FROZEN HELD-OUT TEST")
print("=" * 100)

print(
    "device        :",
    DEVICE,
)

print(
    "cases         :",
    len(truths),
)

print(
    "lambda        :",
    LAMBDA,
)

print(
    "beta          :",
    BETA,
)

print(
    "J             :",
    J,
)

print(
    "iterations    :",
    NITER,
)

print(
    "tau           :",
    TAU,
)

print(
    "sigma         :",
    SIGMA,
)

print(
    "batch         :",
    BATCH,
)

print(
    "Starlet norm² :",
    star.norm2,
)

print(
    "P(k)          :",
    PK_FILE,
)

print(
    "existing      :",
    len(done),
)

print()


# ============================================================
# RUN HELD-OUT TEST
# ============================================================
total_start = time.perf_counter()

for ib in range(
    0,
    len(truths),
    BATCH,
):
    ie = min(
        ib + BATCH,
        len(truths),
    )

    batch_meta = meta[
        ib:ie
    ]

    # If entire batch already done, skip.
    if all(
        item in done
        for item in batch_meta
    ):
        print(
            f"{ie:3d}/480 already complete",
            flush=True,
        )
        continue

    # In a partially completed batch, recompute all;
    # deduplication below keeps only one result.
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

    # Frozen initialization.
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

    t0 = time.perf_counter()

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
        alpha=alpha,
        x0=x0,
    )

    batch_runtime = (
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

    for j, met in enumerate(
        mets
    ):
        idx = ib + j

        c, r, pidx = meta[
            idx
        ]

        mse, nmse, pcc, sr = (
            met
        )

        obs = masks[
            idx
        ].astype(bool)

        # --------------------------------------------
        # SAVE MAPS
        # --------------------------------------------
        rr = r - 10

        rec_map[
            c,
            rr,
            pidx,
        ] = rec_np[j]

        gauss_map[
            c,
            rr,
            pidx,
        ] = xg_np[j]

        sparse_map[
            c,
            rr,
            pidx,
        ] = xs_np[j]

        # --------------------------------------------
        # METRICS
        # --------------------------------------------
        row = {
            "method": "GAUSSIAN_STARLET",

            "c": c,
            "r": r,
            "p": pidx,

            "lambda": LAMBDA,
            "beta": BETA,

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

            "last_rel_change": float(
                rel_np[j]
            ),

            "runtime": float(
                batch_runtime
                / (ie - ib)
            ),

            "f_obs": float(
                masks[idx].mean()
            ),
        }

        rows.append(
            row
        )

    # Flush map files.
    rec_map.flush()
    gauss_map.flush()
    sparse_map.flush()

    # Save metrics after every batch.
    tab_now = pd.DataFrame(
        rows
    )

    tab_now = (
        tab_now
        .drop_duplicates(
            subset=[
                "c",
                "r",
                "p",
            ],
            keep="last",
        )
        .sort_values(
            [
                "c",
                "r",
                "p",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    tab_now.to_csv(
        CSV,
        index=False,
    )

    rows = tab_now.to_dict(
        "records"
    )

    print(
        f"{ie:3d}/480  "
        f"batch={batch_runtime:.3f}s  "
        f"mean_rel={np.mean(rel_np):.3e}",
        flush=True,
    )


# ============================================================
# FINAL METRIC TABLE
# ============================================================
tab = pd.DataFrame(
    rows
)

tab = (
    tab
    .drop_duplicates(
        subset=[
            "c",
            "r",
            "p",
        ],
        keep="last",
    )
    .sort_values(
        [
            "c",
            "r",
            "p",
        ]
    )
    .reset_index(
        drop=True
    )
)

if len(tab) != 480:
    raise RuntimeError(
        f"Expected 480 cases; got {len(tab)}"
    )

tab.to_csv(
    CSV,
    index=False,
)

elapsed_total = (
    time.perf_counter()
    - total_start
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================
summary = pd.DataFrame(
    [
        {
            "method": "GAUSSIAN_STARLET",

            "n": len(tab),

            "nmse_mean": (
                tab["nmse"].mean()
            ),

            "nmse_std": (
                tab["nmse"].std()
            ),

            "nmse_median": (
                tab["nmse"].median()
            ),

            "mse_mean": (
                tab["mse"].mean()
            ),

            "pcc_mean": (
                tab["pcc"].mean()
            ),

            "std_ratio_mean": (
                tab["std_ratio"].mean()
            ),

            "gaussian_std_mean": (
                tab[
                    "std_gaussian"
                ].mean()
            ),

            "sparse_std_mean": (
                tab[
                    "std_sparse"
                ].mean()
            ),

            "rel_change_mean": (
                tab[
                    "last_rel_change"
                ].mean()
            ),

            "rel_change_max": (
                tab[
                    "last_rel_change"
                ].max()
            ),

            "runtime_mean": (
                tab[
                    "runtime"
                ].mean()
            ),
        }
    ]
)

summary.to_csv(
    SUMMARY_CSV,
    index=False,
)

print()
print("=" * 120)
print("GLOBAL TEST SUMMARY")
print("=" * 120)

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
        "c"
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
    )
    .reset_index()
)

byc.to_csv(
    BYCOSMO_CSV,
    index=False,
)

print()
print("=" * 100)
print("BY COSMOLOGY")
print("=" * 100)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# ============================================================
# BOOTSTRAP
# ============================================================
def percentile_ci(x):
    return (
        float(
            np.percentile(
                x,
                2.5,
            )
        ),
        float(
            np.percentile(
                x,
                97.5,
            )
        ),
    )


def paired_block_bootstrap(
    ours,
    base,
    base_name,
):
    """
    Block bootstrap over R10-R29.

    One block contains:
        3 cosmologies x 8 patches = 24 cases.
    """
    a = ours[
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
            "nmse": "nmse_ours",
            "pcc": "pcc_ours",
        }
    )

    b = base[
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    b = b.rename(
        columns={
            "nmse": "nmse_base",
            "pcc": "pcc_base",
        }
    )

    q = b.merge(
        a,
        on=[
            "c",
            "r",
            "p",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(q) != 480:
        raise RuntimeError(
            f"{base_name}: expected 480 matched "
            f"cases, got {len(q)}"
        )

    block = (
        q
        .groupby(
            "r"
        )
        .agg(
            nmse_base=(
                "nmse_base",
                "mean",
            ),

            nmse_ours=(
                "nmse_ours",
                "mean",
            ),

            pcc_base=(
                "pcc_base",
                "mean",
            ),

            pcc_ours=(
                "pcc_ours",
                "mean",
            ),
        )
        .sort_index()
    )

    if len(block) != 20:
        raise RuntimeError(
            f"{base_name}: expected 20 blocks, "
            f"got {len(block)}"
        )

    base_nmse = (
        q["nmse_base"].mean()
    )

    ours_nmse = (
        q["nmse_ours"].mean()
    )

    gain_abs = (
        base_nmse
        - ours_nmse
    )

    gain_pct = (
        100.0
        * gain_abs
        / base_nmse
    )

    pcc_gain = (
        q["pcc_ours"].mean()
        - q["pcc_base"].mean()
    )

    b_nmse = (
        block[
            "nmse_base"
        ].to_numpy()
    )

    o_nmse = (
        block[
            "nmse_ours"
        ].to_numpy()
    )

    b_pcc = (
        block[
            "pcc_base"
        ].to_numpy()
    )

    o_pcc = (
        block[
            "pcc_ours"
        ].to_numpy()
    )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    idx = rng.integers(
        0,
        20,
        size=(
            NBOOT,
            20,
        ),
    )

    bs_base = (
        b_nmse[idx]
        .mean(axis=1)
    )

    bs_ours = (
        o_nmse[idx]
        .mean(axis=1)
    )

    bs_abs = (
        bs_base
        - bs_ours
    )

    bs_pct = (
        100.0
        * bs_abs
        / bs_base
    )

    bs_pcc = (
        o_pcc[idx]
        .mean(axis=1)
        -
        b_pcc[idx]
        .mean(axis=1)
    )

    abs_lo, abs_hi = (
        percentile_ci(
            bs_abs
        )
    )

    pct_lo, pct_hi = (
        percentile_ci(
            bs_pct
        )
    )

    pcc_lo, pcc_hi = (
        percentile_ci(
            bs_pcc
        )
    )

    patch_wins = int(
        np.sum(
            q["nmse_ours"]
            <
            q["nmse_base"]
        )
    )

    block_wins = int(
        np.sum(
            o_nmse
            <
            b_nmse
        )
    )

    return {
        "comparison": (
            "GAUSSIAN_STARLET "
            f"vs {base_name}"
        ),

        "n_cases": 480,
        "n_blocks": 20,

        "base_nmse": float(
            base_nmse
        ),

        "ours_nmse": float(
            ours_nmse
        ),

        "nmse_gain_abs": float(
            gain_abs
        ),

        "nmse_gain_abs_ci_lo": abs_lo,
        "nmse_gain_abs_ci_hi": abs_hi,

        "nmse_gain_pct": float(
            gain_pct
        ),

        "nmse_gain_pct_ci_lo": pct_lo,
        "nmse_gain_pct_ci_hi": pct_hi,

        "pcc_gain": float(
            pcc_gain
        ),

        "pcc_gain_ci_lo": pcc_lo,
        "pcc_gain_ci_hi": pcc_hi,

        "patch_wins": patch_wins,
        "patch_total": 480,

        "block_wins": block_wins,
        "block_total": 20,
    }


# ============================================================
# LOAD COMPARATORS
# ============================================================
pairwise_rows = []

# ------------------------------------------------------------
# OUR EXISTING METHODS
# ------------------------------------------------------------
OURS_FILE = Path(
    str(PATHS.work_root / 'test/square_ours_metrics.csv')
)

if OURS_FILE.exists():
    base = pd.read_csv(
        OURS_FILE
    )

    for method in [
        "KS",
        "STARLET_NOMASK",
        "STARLET_MASK",
    ]:
        q = base[
            base["method"]
            == method
        ].copy()

        if len(q) == 480:
            pairwise_rows.append(
                paired_block_bootstrap(
                    tab,
                    q,
                    method,
                )
            )
        else:
            print(
                f"Skipping {method}: "
                f"found {len(q)} cases"
            )

else:
    print(
        "Missing comparator:",
        OURS_FILE,
    )


# ------------------------------------------------------------
# MCALENS
# ------------------------------------------------------------
MC_FILE = Path(
    str(PATHS.work_root / 'test/mcalens/mcalens_test_metrics.csv')
)

if MC_FILE.exists():
    mc = pd.read_csv(
        MC_FILE
    )

    if len(mc) == 480:
        pairwise_rows.append(
            paired_block_bootstrap(
                tab,
                mc,
                "MCALENS",
            )
        )

    else:
        print(
            "MCALens metric count:",
            len(mc),
        )

else:
    print(
        "MCALens CSV not found:",
        MC_FILE,
    )


# ============================================================
# OPTIONAL GLIMPSE DISCOVERY
# ============================================================
glimpse_candidates = [
    Path(
        str(PATHS.work_root / 'test/glimpse_test_metrics.csv')
    ),

    Path(
        str(PATHS.work_root / 'test/glimpse_metrics.csv')
    ),

    Path(
        str(PATHS.work_root / 'test/fair_glimpse_test_metrics.csv')
    ),
]

for gf in glimpse_candidates:
    if not gf.exists():
        continue

    try:
        gl = pd.read_csv(
            gf
        )

        required = {
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        }

        if (
            required
            <= set(gl.columns)
            and len(gl) == 480
        ):
            pairwise_rows.append(
                paired_block_bootstrap(
                    tab,
                    gl,
                    "GLIMPSE",
                )
            )

            print(
                "Found GLIMPSE:",
                gf,
            )

            break

    except Exception as exc:
        print(
            "Could not use GLIMPSE file:",
            gf,
            repr(exc),
        )


# ============================================================
# PAIRWISE TABLE
# ============================================================
if pairwise_rows:
    pair = pd.DataFrame(
        pairwise_rows
    )

    pair.to_csv(
        PAIRWISE_CSV,
        index=False,
    )

    print()
    print("=" * 150)
    print("PAIRED REALIZATION-BLOCK BOOTSTRAP")
    print("=" * 150)

    print(
        pair.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )


# ============================================================
# SIMPLE GLOBAL REFERENCE TABLE
# ============================================================
print()
print("=" * 100)
print("GLOBAL MATCHED TEST REFERENCE")
print("=" * 100)

print(
    f"{'GAUSSIAN_STARLET':20s} "
    f"NMSE={tab.nmse.mean():.9f}  "
    f"PCC={tab.pcc.mean():.6f}  "
    f"std={tab.std_ratio.mean():.6f}"
)

if OURS_FILE.exists():
    oo = pd.read_csv(
        OURS_FILE
    )

    for method in [
        "KS",
        "STARLET_NOMASK",
        "STARLET_MASK",
    ]:
        q = oo[
            oo["method"]
            == method
        ]

        if len(q):
            print(
                f"{method:20s} "
                f"NMSE={q.nmse.mean():.9f}  "
                f"PCC={q.pcc.mean():.6f}"
            )

if MC_FILE.exists():
    q = pd.read_csv(
        MC_FILE
    )

    print(
        f"{'MCALENS':20s} "
        f"NMSE={q.nmse.mean():.9f}  "
        f"PCC={q.pcc.mean():.6f}  "
        f"std={q.std_ratio.mean():.6f}"
    )

print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(CSV)
print(SUMMARY_CSV)
print(BYCOSMO_CSV)
print(PAIRWISE_CSV)

print(REC_FILE)
print(GAUSS_FILE)
print(SPARSE_FILE)

print()
print(
    "Total new wall time [s]:",
    elapsed_total,
)

print()
print("FROZEN HELD-OUT TEST COMPLETE")
