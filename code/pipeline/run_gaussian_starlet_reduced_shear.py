#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frozen Gaussian+Starlet reduced-shear stress test
=================================================

PURPOSE
-------
Test the CURRENT Gaussian+Starlet reconstruction under

    g = gamma / (1 - kappa)

without retuning any reconstruction hyperparameter.

The exact same shape-noise realization already present in the
nominal FAIR test observation is reused:

    n = y_nominal - A(kappa)

and

    y_RS = M [ A(kappa)/(1-kappa) + n ].

This therefore isolates the reduced-shear forward-model mismatch.

Frozen reconstruction parameters
--------------------------------
lambda      = 0.05
beta        = 0.001
Starlet J   = 4
alpha       = [1, 0.225148, 0.096029, 0.046286]

PDHG:
tau         = 0.5
sigma       = 0.5
theta       = 1.0
iterations  = 6000

Gaussian prior:
same validation-derived P_kappa used by the published
Gaussian+Starlet benchmark.

IMPORTANT
---------
Before running the stress test, this script reconstructs three nominal
held-out patches and compares them with the already frozen G+S maps.

If the solver implementation here does not reproduce the frozen maps
to the requested tolerance, the script ABORTS.

No test-time tuning is performed.

Outputs are checkpointed one held-out realization at a time:
R10 ... R29.
"""
from project_paths import PATHS

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS / IMPORTS
# =============================================================================

ROOT = Path(
    str(PATHS.code_root)
)

sys.path.insert(
    0,
    str(ROOT),
)

from weak_lensing_pipeline import WeakLensingOperator


DATA = PATHS.data_root

MASK_FILE = (
    DATA
    / "WIDE12H_bin2_2arcmin_mask.npy"
)

FAIR_ROOT = Path(
    str(PATHS.work_root / 'test/fair_npz')
)

PK_FILE = Path(
    str(PATHS.prior_file)
)

GS_NOMINAL_FILE = Path(
    str(PATHS.work_root / 'test/starlet_gaussian_prior/gaussian_starlet_test_reconstruction.npy')
)

OUT = Path(
    str(PATHS.work_root / 'test/gaussian_starlet_reduced_shear')
)

BLOCK_DIR = (
    OUT
    / "blocks"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BLOCK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

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

TAU = 0.5
SIGMA = 0.5
THETA = 1.0

NITER = 6000

TEST_REALIZATIONS = list(
    range(
        10,
        30,
    )
)

PATCH_ROWS = [
    (
        p * N,
        (p + 1) * N,
    )
    for p in range(8)
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = torch.float32
CDTYPE = torch.complex64


# =============================================================================
# PREFLIGHT TOLERANCES
# =============================================================================

# Full-map relative reconstruction difference.
PREFLIGHT_MAP_REL_TOL = 5.0e-3

# Difference in individual field metrics.
PREFLIGHT_METRIC_TOL = 1.0e-3


# =============================================================================
# INPUT CHECKS
# =============================================================================

for fn in [
    MASK_FILE,
    PK_FILE,
    GS_NOMINAL_FILE,
]:

    if not fn.exists():
        raise FileNotFoundError(
            fn
        )


if not FAIR_ROOT.exists():

    raise FileNotFoundError(
        FAIR_ROOT
    )


print()
print("=" * 120)
print("GAUSSIAN+STARLET REDUCED-SHEAR STRESS TEST")
print("=" * 120)

print(
    "device       :",
    DEVICE
)

print(
    "lambda       :",
    LAMBDA
)

print(
    "beta         :",
    BETA
)

print(
    "Starlet J    :",
    J
)

print(
    "alpha        :",
    ALPHA.tolist()
)

print(
    "PDHG         :",
    f"tau={TAU}, sigma={SIGMA}, "
    f"theta={THETA}, niter={NITER}"
)


# =============================================================================
# LOAD MASK / FROZEN NOMINAL MAPS
# =============================================================================

mask_full = np.load(
    MASK_FILE
).astype(bool)


MASK_PATCH = np.stack(
    [
        mask_full[
            r0:r1,
            :
        ]
        for r0, r1 in PATCH_ROWS
    ],
    axis=0,
)


GS_NOMINAL = np.load(
    GS_NOMINAL_FILE,
    mmap_mode="r",
)

GS_NOMINAL = np.asarray(
    GS_NOMINAL
)


if GS_NOMINAL.shape == (
    3,
    30,
    8,
    N,
    N,
):

    GS_NOMINAL = GS_NOMINAL[
        :,
        10:30,
        :,
        :,
        :,
    ]


elif GS_NOMINAL.shape == (
    480,
    N,
    N,
):

    GS_NOMINAL = GS_NOMINAL.reshape(
        3,
        20,
        8,
        N,
        N,
    )


if GS_NOMINAL.shape != (
    3,
    20,
    8,
    N,
    N,
):

    raise ValueError(
        "Unexpected frozen G+S shape: "
        f"{GS_NOMINAL.shape}"
    )


# =============================================================================
# WEAK-LENSING OPERATOR
# =============================================================================

OP = WeakLensingOperator(
    N,
    N,
    DEVICE,
)


def A_batch(
    x,
):
    """
    x:
        [B,H,W], real

    returns:
        [B,H,W], complex
    """

    return torch.vmap(
        OP.forward
    )(
        x
    )


def AH_batch(
    y,
):
    """
    y:
        [B,H,W], complex

    returns:
        [B,H,W], real
    """

    z = torch.vmap(
        OP.adjoint
    )(
        y
    )

    if torch.is_complex(
        z
    ):
        z = z.real

    return z


# =============================================================================
# P_KAPPA
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

    # Remove mean / mass-sheet mode.
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

    nk = len(pk)

    ky = (
        np.arange(H)
        -
        H / 2.0
    )

    kx = (
        np.arange(W)
        -
        W / 2.0
    )

    KY, KX = np.meshgrid(
        ky,
        kx,
        indexing="ij",
    )

    ir = np.sqrt(
        KX**2
        +
        KY**2
    ).astype(
        np.int64
    )

    ir[
        ir > nk
    ] = nk

    return np.fft.ifftshift(
        ext[
            ir
        ]
    )


pk1d = np.load(
    PK_FILE
).astype(
    np.float64
)


PK = torch.as_tensor(
    make_pk_map(
        pk1d,
        N,
        N,
    ),
    dtype=DTYPE,
    device=DEVICE,
)


PK_SUPPORT = (
    PK > 0
).to(
    DTYPE
)


# =============================================================================
# EXACT PERIODIC B3-SPLINE STARLET FOURIER FILTERS
# =============================================================================

B3 = np.array(
    [
        1.0,
        4.0,
        6.0,
        4.0,
        1.0,
    ],
    dtype=np.float64,
) / 16.0


def dilated_b3_transfer(
    n,
    step,
):

    kernel_1d = np.zeros(
        n,
        dtype=np.float64,
    )

    offsets = np.array(
        [
            -2,
            -1,
            0,
            1,
            2,
        ],
        dtype=int,
    ) * step

    for coeff, offset in zip(
        B3,
        offsets,
    ):

        kernel_1d[
            offset % n
        ] += coeff


    eig_1d = np.fft.fft(
        kernel_1d
    )


    return (
        eig_1d[:, None]
        *
        eig_1d[None, :]
    )


starlet_w = []

cumulative = np.ones(
    (
        N,
        N,
    ),
    dtype=np.complex128,
)


for j in range(
    J
):

    Hj = dilated_b3_transfer(
        N,
        2**j,
    )

    Wj = (
        cumulative
        *
        (
            1.0
            -
            Hj
        )
    )

    starlet_w.append(
        Wj
    )

    cumulative = (
        cumulative
        *
        Hj
    )


STARLET_W = torch.as_tensor(
    np.stack(
        starlet_w,
        axis=0,
    ),
    dtype=CDTYPE,
    device=DEVICE,
)


ALPHA_T = torch.as_tensor(
    ALPHA,
    dtype=DTYPE,
    device=DEVICE,
).view(
    1,
    J,
    1,
    1,
)


# =============================================================================
# STARLET ANALYSIS / ADJOINT
# =============================================================================

def starlet_analysis(
    x,
):
    """
    x:
        [B,H,W]

    returns:
        [B,J,H,W]
    """

    X = torch.fft.fft2(
        x,
        norm="ortho",
    )

    W = (
        X[:, None, :, :]
        *
        STARLET_W[
            None,
            :,
            :,
            :
        ]
    )

    return torch.fft.ifft2(
        W,
        norm="ortho",
    ).real


def starlet_adjoint(
    q,
):
    """
    q:
        [B,J,H,W]

    returns:
        [B,H,W]
    """

    Q = torch.fft.fft2(
        q,
        norm="ortho",
    )

    # Filters are real-space symmetric, so use conjugate
    # explicitly even though numerical differences are tiny.
    accum = torch.sum(
        Q
        *
        torch.conj(
            STARLET_W
        )[
            None,
            :,
            :,
            :
        ],
        dim=1,
    )

    return torch.fft.ifft2(
        accum,
        norm="ortho",
    ).real


# =============================================================================
# GAUSSIAN PRIOR PROX
# =============================================================================

def gaussian_prox(
    x,
):
    """
    prox of

        tau * beta/2 *
        sum_{Pk>0} |Fx|^2 / Pk

    with exact zero outside the P_kappa support.
    """

    X = torch.fft.fft2(
        x,
        norm="ortho",
    )

    factor = torch.where(
        PK > 0,
        PK
        /
        (
            PK
            +
            TAU
            *
            BETA
        ),
        torch.zeros_like(
            PK
        ),
    )

    return torch.fft.ifft2(
        X
        *
        factor[
            None,
            :,
            :
        ],
        norm="ortho",
    ).real


# =============================================================================
# PDHG SOLVER
# =============================================================================

@torch.no_grad()
def solve_gaussian_starlet(
    y,
    mask,
):
    """
    Solve

      min_{xG,xS}
          1/2 ||M[A(xG+xS)-y]||_2^2
        + beta/2 sum |F xG|^2/Pk
        + lambda sum_j alpha_j ||W_j xS||_1

    Parameters
    ----------
    y:
        [B,H,W] complex observed shear

    mask:
        [B,H,W] real {0,1}
    """

    B = y.shape[0]

    xg = torch.zeros(
        (
            B,
            N,
            N,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )

    xs = torch.zeros_like(
        xg
    )

    xg_bar = xg.clone()
    xs_bar = xs.clone()

    p = torch.zeros(
        (
            B,
            N,
            N,
        ),
        dtype=CDTYPE,
        device=DEVICE,
    )

    q = torch.zeros(
        (
            B,
            J,
            N,
            N,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )


    threshold = (
        LAMBDA
        *
        ALPHA_T
    )


    for it in range(
        1,
        NITER + 1,
    ):

        # ---------------------------------------------------------
        # DATA DUAL
        #
        # prox_{sigma f*}, f(u)=1/2||u-y||^2
        # ---------------------------------------------------------

        pred = A_batch(
            xg_bar
            +
            xs_bar
        )

        p = (
            p
            +
            SIGMA
            *
            (
                mask
                *
                pred
                -
                y
            )
        ) / (
            1.0
            +
            SIGMA
        )

        # Restrict dual variable to observed pixels.
        p = (
            mask
            *
            p
        )


        # ---------------------------------------------------------
        # STARLET L1 DUAL
        # ---------------------------------------------------------

        wx = starlet_analysis(
            xs_bar
        )

        q = (
            q
            +
            SIGMA
            *
            wx
        )

        q = torch.maximum(
            torch.minimum(
                q,
                threshold,
            ),
            -threshold,
        )


        # ---------------------------------------------------------
        # PRIMAL GRADIENT
        # ---------------------------------------------------------

        grad_data = AH_batch(
            mask
            *
            p
        )

        grad_sparse = starlet_adjoint(
            q
        )


        xg_old = xg
        xs_old = xs


        xg = gaussian_prox(
            xg
            -
            TAU
            *
            grad_data
        )

        xs = (
            xs
            -
            TAU
            *
            (
                grad_data
                +
                grad_sparse
            )
        )


        # ---------------------------------------------------------
        # EXTRAPOLATION
        # ---------------------------------------------------------

        xg_bar = (
            xg
            +
            THETA
            *
            (
                xg
                -
                xg_old
            )
        )

        xs_bar = (
            xs
            +
            THETA
            *
            (
                xs
                -
                xs_old
            )
        )


    return (
        xg
        +
        xs
    )


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

    r -= r.mean()
    t -= t.mean()

    nmse = (
        np.sum(
            (
                r
                -
                t
            )**2
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
            +
            1e-30
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
            +
            1e-30
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


# =============================================================================
# LOAD ONE PATCH
# =============================================================================

def load_patch(
    c,
    r,
    p,
):

    fn = (
        FAIR_ROOT
        / f"C{c}_R{r}.npz"
    )

    if not fn.exists():

        raise FileNotFoundError(
            fn
        )


    with np.load(
        fn
    ) as z:

        if "truth" not in z:

            raise KeyError(
                f"{fn}: no 'truth'"
            )

        if "gamma" not in z:

            raise KeyError(
                f"{fn}: no 'gamma'"
            )

        truth_full = np.asarray(
            z[
                "truth"
            ],
            dtype=np.float32,
        )

        gamma_full = np.asarray(
            z[
                "gamma"
            ],
            dtype=np.complex64,
        )


    r0, r1 = PATCH_ROWS[
        p
    ]

    truth = truth_full[
        r0:r1,
        :
    ].copy()

    gamma_obs = gamma_full[
        r0:r1,
        :
    ].copy()

    mask = MASK_PATCH[
        p
    ].copy()

    return (
        truth,
        gamma_obs,
        mask,
    )


# =============================================================================
# CONSTRUCT NOMINAL + REDUCED-SHEAR OBSERVATIONS
# =============================================================================

@torch.no_grad()
def construct_observations(
    truths_np,
    observed_np,
    masks_np,
):

    truth_t = torch.as_tensor(
        truths_np,
        dtype=DTYPE,
        device=DEVICE,
    )

    obs_t = torch.as_tensor(
        observed_np,
        dtype=CDTYPE,
        device=DEVICE,
    )

    mask_t = torch.as_tensor(
        masks_np.astype(
            np.float32
        ),
        dtype=DTYPE,
        device=DEVICE,
    )


    gamma_true = A_batch(
        truth_t
    )


    # Existing exact noise realization from the nominal benchmark.
    noise = (
        obs_t
        -
        gamma_true
    )

    noise = (
        mask_t
        *
        noise
    )


    denom = (
        1.0
        -
        truth_t
    )


    min_abs_denom = float(
        torch.min(
            torch.abs(
                denom[
                    mask_t
                    > 0
                ]
            )
        ).item()
    )


    if min_abs_denom < 0.05:

        raise RuntimeError(
            "Reduced-shear denominator too close to zero: "
            f"{min_abs_denom}"
        )


    g_true = (
        gamma_true
        /
        denom.to(
            CDTYPE
        )
    )


    y_nominal = (
        mask_t
        *
        obs_t
    )


    y_rs = (
        mask_t
        *
        (
            g_true
            +
            noise
        )
    )


    return (
        truth_t,
        mask_t,
        gamma_true,
        g_true,
        noise,
        y_nominal,
        y_rs,
    )


# =============================================================================
# PREFLIGHT
# =============================================================================

print()
print("=" * 120)
print("PREFLIGHT — REPRODUCE FROZEN NOMINAL G+S")
print("=" * 120)


PREFLIGHT_CASES = [
    (
        0,
        10,
        0,
    ),
    (
        1,
        10,
        4,
    ),
    (
        2,
        10,
        7,
    ),
]


truths = []
observed = []
masks = []
refs = []


for c, r, p in PREFLIGHT_CASES:

    truth, gamma_obs, mask = load_patch(
        c,
        r,
        p,
    )

    truths.append(
        truth
    )

    observed.append(
        gamma_obs
    )

    masks.append(
        mask
    )

    refs.append(
        np.asarray(
            GS_NOMINAL[
                c,
                r - 10,
                p,
            ],
            dtype=np.float32,
        )
    )


truths = np.stack(
    truths
)

observed = np.stack(
    observed
)

masks = np.stack(
    masks
)

refs = np.stack(
    refs
)


(
    truth_t,
    mask_t,
    gamma_true_t,
    g_true_t,
    noise_t,
    y_nominal_t,
    y_rs_t,
) = construct_observations(
    truths,
    observed,
    masks,
)


t0 = time.perf_counter()


rec_nominal_t = solve_gaussian_starlet(
    y_nominal_t,
    mask_t,
)


rec_nominal = (
    rec_nominal_t
    .cpu()
    .numpy()
)


print(
    f"preflight runtime: "
    f"{time.perf_counter()-t0:.2f} s"
)


preflight_ok = True


for i, (
    c,
    r,
    p,
) in enumerate(
    PREFLIGHT_CASES
):

    ref = refs[
        i
    ]

    rec = rec_nominal[
        i
    ]

    rel = (
        np.linalg.norm(
            rec
            -
            ref
        )
        /
        (
            np.linalg.norm(
                ref
            )
            +
            1e-30
        )
    )


    m_rec = metrics(
        rec,
        truths[
            i
        ],
        masks[
            i
        ],
    )

    m_ref = metrics(
        ref,
        truths[
            i
        ],
        masks[
            i
        ],
    )


    mdiff = np.max(
        np.abs(
            np.asarray(
                m_rec
            )
            -
            np.asarray(
                m_ref
            )
        )
    )


    print()
    print(
        f"C{c} R{r} P{p}"
    )

    print(
        f"  map relative difference = "
        f"{rel:.8e}"
    )

    print(
        "  reconstructed metrics   =",
        tuple(
            f"{x:.9f}"
            for x in m_rec
        )
    )

    print(
        "  frozen metrics          =",
        tuple(
            f"{x:.9f}"
            for x in m_ref
        )
    )

    print(
        f"  max metric difference   = "
        f"{mdiff:.8e}"
    )


    if (
        rel
        >
        PREFLIGHT_MAP_REL_TOL
        or
        mdiff
        >
        PREFLIGHT_METRIC_TOL
    ):

        preflight_ok = False


if not preflight_ok:

    print()
    print("=" * 120)
    print("PREFLIGHT FAILED")
    print("=" * 120)

    raise RuntimeError(
        "This local PDHG implementation does not reproduce "
        "the frozen Gaussian+Starlet maps closely enough. "
        "Reduced-shear reconstruction was NOT started."
    )


print()
print("=" * 120)
print("PREFLIGHT PASSED")
print("=" * 120)

print(
    "Proceeding with frozen reduced-shear test."
)


# Free preflight tensors.
del (
    rec_nominal_t,
    rec_nominal,
    truth_t,
    mask_t,
    gamma_true_t,
    g_true_t,
    noise_t,
    y_nominal_t,
    y_rs_t,
)

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()


# =============================================================================
# REDUCED-SHEAR HELD-OUT TEST
#
# Checkpoint unit:
#   one realization = 3 cosmologies x 8 patches = 24 maps
# =============================================================================

for r in TEST_REALIZATIONS:

    block_file = (
        BLOCK_DIR
        / f"R{r:02d}.npz"
    )


    if block_file.exists():

        try:

            with np.load(
                block_file
            ) as z:

                if (
                    "rec_rs"
                    in z
                    and
                    z[
                        "rec_rs"
                    ].shape
                    ==
                    (
                        3,
                        8,
                        N,
                        N,
                    )
                ):

                    print(
                        f"R{r}: checkpoint exists — SKIP"
                    )

                    continue

        except Exception:
            pass


    print()
    print("=" * 120)
    print(
        f"REDUCED SHEAR — R{r}"
    )
    print("=" * 120)


    truths = []
    observed = []
    masks = []
    indices = []


    for c in range(3):

        for p in range(8):

            truth, gamma_obs, mask = load_patch(
                c,
                r,
                p,
            )

            truths.append(
                truth
            )

            observed.append(
                gamma_obs
            )

            masks.append(
                mask
            )

            indices.append(
                (
                    c,
                    p,
                )
            )


    truths = np.stack(
        truths
    )

    observed = np.stack(
        observed
    )

    masks = np.stack(
        masks
    )


    (
        truth_t,
        mask_t,
        gamma_true_t,
        g_true_t,
        noise_t,
        y_nominal_t,
        y_rs_t,
    ) = construct_observations(
        truths,
        observed,
        masks,
    )


    # ---------------------------------------------------------
    # Diagnostics of the physical perturbation
    # ---------------------------------------------------------

    mismatch = []

    noise_sigma_real = []
    noise_sigma_imag = []


    gamma_true_np = (
        gamma_true_t
        .cpu()
        .numpy()
    )

    g_true_np = (
        g_true_t
        .cpu()
        .numpy()
    )

    noise_np = (
        noise_t
        .cpu()
        .numpy()
    )


    for i in range(
        len(
            indices
        )
    ):

        m = masks[
            i
        ]

        mismatch.append(
            np.linalg.norm(
                (
                    g_true_np[
                        i
                    ]
                    -
                    gamma_true_np[
                        i
                    ]
                )[
                    m
                ]
            )
            /
            (
                np.linalg.norm(
                    gamma_true_np[
                        i
                    ][
                        m
                    ]
                )
                +
                1e-30
            )
        )

        noise_sigma_real.append(
            np.std(
                noise_np[
                    i
                ].real[
                    m
                ]
            )
        )

        noise_sigma_imag.append(
            np.std(
                noise_np[
                    i
                ].imag[
                    m
                ]
            )
        )


    print(
        "mean |g-gamma|/|gamma| =",
        f"{np.mean(mismatch):.6f}"
    )

    print(
        "noise sigma real       =",
        f"{np.mean(noise_sigma_real):.8f}"
    )

    print(
        "noise sigma imag       =",
        f"{np.mean(noise_sigma_imag):.8f}"
    )


    # ---------------------------------------------------------
    # Frozen reconstruction
    # ---------------------------------------------------------

    t0 = time.perf_counter()


    rec_rs_t = solve_gaussian_starlet(
        y_rs_t,
        mask_t,
    )


    runtime = (
        time.perf_counter()
        -
        t0
    )


    rec_rs_flat = (
        rec_rs_t
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    rec_rs = np.empty(
        (
            3,
            8,
            N,
            N,
        ),
        dtype=np.float32,
    )


    mismatch_arr = np.empty(
        (
            3,
            8,
        ),
        dtype=np.float32,
    )


    noise_real_arr = np.empty(
        (
            3,
            8,
        ),
        dtype=np.float32,
    )


    noise_imag_arr = np.empty(
        (
            3,
            8,
        ),
        dtype=np.float32,
    )


    for i, (
        c,
        p,
    ) in enumerate(
        indices
    ):

        rec_rs[
            c,
            p,
        ] = rec_rs_flat[
            i
        ]

        mismatch_arr[
            c,
            p,
        ] = mismatch[
            i
        ]

        noise_real_arr[
            c,
            p,
        ] = noise_sigma_real[
            i
        ]

        noise_imag_arr[
            c,
            p,
        ] = noise_sigma_imag[
            i
        ]


    np.savez_compressed(
        block_file,
        rec_rs=rec_rs,
        mismatch_relative=mismatch_arr,
        noise_sigma_real=noise_real_arr,
        noise_sigma_imag=noise_imag_arr,
        runtime_s=np.float64(
            runtime
        ),
    )


    print(
        f"runtime = {runtime:.2f} s"
    )

    print(
        "saved  =",
        block_file
    )


    del (
        truth_t,
        mask_t,
        gamma_true_t,
        g_true_t,
        noise_t,
        y_nominal_t,
        y_rs_t,
        rec_rs_t,
        rec_rs_flat,
    )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# AGGREGATE ALL 20 CHECKPOINTS
# =============================================================================

print()
print("=" * 120)
print("AGGREGATING")
print("=" * 120)


RS = np.empty(
    (
        3,
        20,
        8,
        N,
        N,
    ),
    dtype=np.float32,
)


MISMATCH = np.empty(
    (
        3,
        20,
        8,
    ),
    dtype=np.float32,
)


NOISE_REAL = np.empty_like(
    MISMATCH
)

NOISE_IMAG = np.empty_like(
    MISMATCH
)


for ri, r in enumerate(
    TEST_REALIZATIONS
):

    block_file = (
        BLOCK_DIR
        / f"R{r:02d}.npz"
    )

    if not block_file.exists():

        raise FileNotFoundError(
            block_file
        )


    with np.load(
        block_file
    ) as z:

        RS[
            :,
            ri,
            :,
        ] = z[
            "rec_rs"
        ]

        MISMATCH[
            :,
            ri,
            :,
        ] = z[
            "mismatch_relative"
        ]

        NOISE_REAL[
            :,
            ri,
            :,
        ] = z[
            "noise_sigma_real"
        ]

        NOISE_IMAG[
            :,
            ri,
            :,
        ] = z[
            "noise_sigma_imag"
        ]


np.save(
    OUT
    / "gaussian_starlet_reduced_shear_test.npy",
    RS,
)


# =============================================================================
# CASE METRICS
# =============================================================================

rows = []


for c in range(3):

    for ri, r in enumerate(
        TEST_REALIZATIONS
    ):

        for p in range(8):

            truth, _, mask = load_patch(
                c,
                r,
                p,
            )


            nominal = np.asarray(
                GS_NOMINAL[
                    c,
                    ri,
                    p,
                ],
                dtype=np.float32,
            )


            reduced = RS[
                c,
                ri,
                p,
            ]


            (
                n_nmse,
                n_pcc,
                n_std,
            ) = metrics(
                nominal,
                truth,
                mask,
            )


            (
                rs_nmse,
                rs_pcc,
                rs_std,
            ) = metrics(
                reduced,
                truth,
                mask,
            )


            rows.append(
                {
                    "c":
                        c,

                    "r":
                        r,

                    "p":
                        p,

                    "f_obs":
                        float(
                            mask.mean()
                        ),

                    "reduced_shear_mismatch_relative":
                        float(
                            MISMATCH[
                                c,
                                ri,
                                p,
                            ]
                        ),

                    "noise_sigma_real":
                        float(
                            NOISE_REAL[
                                c,
                                ri,
                                p,
                            ]
                        ),

                    "noise_sigma_imag":
                        float(
                            NOISE_IMAG[
                                c,
                                ri,
                                p,
                            ]
                        ),

                    "nominal_nmse":
                        n_nmse,

                    "rs_nmse":
                        rs_nmse,

                    "delta_nmse_rs_minus_nominal":
                        rs_nmse
                        -
                        n_nmse,

                    "relative_nmse_change_pct":
                        100.0
                        *
                        (
                            rs_nmse
                            -
                            n_nmse
                        )
                        /
                        n_nmse,

                    "nominal_pcc":
                        n_pcc,

                    "rs_pcc":
                        rs_pcc,

                    "delta_pcc_rs_minus_nominal":
                        rs_pcc
                        -
                        n_pcc,

                    "nominal_std_ratio":
                        n_std,

                    "rs_std_ratio":
                        rs_std,

                    "delta_std_rs_minus_nominal":
                        rs_std
                        -
                        n_std,
                }
            )


df = pd.DataFrame(
    rows
)


df.to_csv(
    OUT
    / "gaussian_starlet_reduced_shear_cases.csv",
    index=False,
)


# =============================================================================
# FROZEN NOMINAL SANITY
# =============================================================================

print()
print("=" * 120)
print("NOMINAL FROZEN SANITY")
print("=" * 120)


nominal_expected = {
    "nmse":
        0.642805480,

    "pcc":
        0.59660041,

    "std":
        0.58345016,
}


nominal_measured = {
    "nmse":
        df[
            "nominal_nmse"
        ].mean(),

    "pcc":
        df[
            "nominal_pcc"
        ].mean(),

    "std":
        df[
            "nominal_std_ratio"
        ].mean(),
}


for key in nominal_expected:

    print(
        f"{key:<8} "
        f"{nominal_measured[key]:.9f} "
        f"expected={nominal_expected[key]:.9f}"
    )


# =============================================================================
# POINT ESTIMATES
# =============================================================================

summary = {
    "n_cases":
        len(
            df
        ),

    "n_realization_blocks":
        20,

    "mean_reduced_shear_mismatch_relative":
        df[
            "reduced_shear_mismatch_relative"
        ].mean(),

    "median_reduced_shear_mismatch_relative":
        df[
            "reduced_shear_mismatch_relative"
        ].median(),

    "max_reduced_shear_mismatch_relative":
        df[
            "reduced_shear_mismatch_relative"
        ].max(),

    "mean_noise_sigma_real":
        df[
            "noise_sigma_real"
        ].mean(),

    "mean_noise_sigma_imag":
        df[
            "noise_sigma_imag"
        ].mean(),

    "nominal_nmse":
        df[
            "nominal_nmse"
        ].mean(),

    "reduced_shear_nmse":
        df[
            "rs_nmse"
        ].mean(),

    "delta_nmse":
        df[
            "delta_nmse_rs_minus_nominal"
        ].mean(),

    "relative_nmse_change_pct":
        100.0
        *
        (
            df[
                "rs_nmse"
            ].mean()
            -
            df[
                "nominal_nmse"
            ].mean()
        )
        /
        df[
            "nominal_nmse"
        ].mean(),

    "nominal_pcc":
        df[
            "nominal_pcc"
        ].mean(),

    "reduced_shear_pcc":
        df[
            "rs_pcc"
        ].mean(),

    "delta_pcc":
        df[
            "delta_pcc_rs_minus_nominal"
        ].mean(),

    "nominal_std_ratio":
        df[
            "nominal_std_ratio"
        ].mean(),

    "reduced_shear_std_ratio":
        df[
            "rs_std_ratio"
        ].mean(),

    "delta_std_ratio":
        df[
            "delta_std_rs_minus_nominal"
        ].mean(),
}


# =============================================================================
# PAIRED REALIZATION-BLOCK BOOTSTRAP
# =============================================================================

N_BOOT = 10000
BOOT_SEED = 20260930

rng = np.random.default_rng(
    BOOT_SEED
)


def paired_summary(
    d,
):

    nominal_nmse = d[
        "nominal_nmse"
    ].mean()

    rs_nmse = d[
        "rs_nmse"
    ].mean()


    return {
        "delta_nmse":
            rs_nmse
            -
            nominal_nmse,

        "relative_nmse_change_pct":
            100.0
            *
            (
                rs_nmse
                -
                nominal_nmse
            )
            /
            nominal_nmse,

        "delta_pcc":
            d[
                "rs_pcc"
            ].mean()
            -
            d[
                "nominal_pcc"
            ].mean(),

        "delta_std_ratio":
            d[
                "rs_std_ratio"
            ].mean()
            -
            d[
                "nominal_std_ratio"
            ].mean(),
    }


point = paired_summary(
    df
)


boot = {
    key:
        np.empty(
            N_BOOT,
            dtype=np.float64,
        )
    for key in point
}


RIDS = np.arange(
    10,
    30,
)


for ib in range(
    N_BOOT
):

    sampled = rng.choice(
        RIDS,
        size=len(
            RIDS
        ),
        replace=True,
    )

    pieces = []


    for boot_id, r in enumerate(
        sampled
    ):

        tmp = df[
            df[
                "r"
            ]
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


    s = paired_summary(
        b
    )


    for key in boot:

        boot[
            key
        ][
            ib
        ] = s[
            key
        ]


boot_rows = []


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


    boot_rows.append(
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
                    values
                    > 0
                ),

            "n_blocks":
                20,

            "n_cases":
                480,

            "n_boot":
                N_BOOT,
        }
    )


boot_df = pd.DataFrame(
    boot_rows
)


boot_df.to_csv(
    OUT
    / "gaussian_starlet_reduced_shear_bootstrap.csv",
    index=False,
)


# =============================================================================
# BY COSMOLOGY
# =============================================================================

by_cosmo = (
    df
    .groupby(
        "c"
    )
    .agg(
        n=(
            "rs_nmse",
            "size",
        ),

        mismatch=(
            "reduced_shear_mismatch_relative",
            "mean",
        ),

        nominal_nmse=(
            "nominal_nmse",
            "mean",
        ),

        rs_nmse=(
            "rs_nmse",
            "mean",
        ),

        delta_nmse=(
            "delta_nmse_rs_minus_nominal",
            "mean",
        ),

        nominal_pcc=(
            "nominal_pcc",
            "mean",
        ),

        rs_pcc=(
            "rs_pcc",
            "mean",
        ),
    )
    .reset_index()
)


by_cosmo.to_csv(
    OUT
    / "gaussian_starlet_reduced_shear_by_cosmology.csv",
    index=False,
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    [
        summary
    ]
)


summary_df.to_csv(
    OUT
    / "gaussian_starlet_reduced_shear_summary.csv",
    index=False,
)


# =============================================================================
# REPORT
# =============================================================================

print()
print("=" * 150)
print("GAUSSIAN+STARLET REDUCED-SHEAR SUMMARY")
print("=" * 150)


print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print()
print("=" * 150)
print("PAIRED REALIZATION-BLOCK BOOTSTRAP")
print("=" * 150)


print(
    boot_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print()
print("=" * 150)
print("BY COSMOLOGY")
print("=" * 150)


print(
    by_cosmo.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.9f}",
    )
)


print()
print("=" * 150)
print("FILES")
print("=" * 150)


for fn in [
    "gaussian_starlet_reduced_shear_test.npy",
    "gaussian_starlet_reduced_shear_cases.csv",
    "gaussian_starlet_reduced_shear_summary.csv",
    "gaussian_starlet_reduced_shear_bootstrap.csv",
    "gaussian_starlet_reduced_shear_by_cosmology.csv",
]:

    print(
        OUT
        / fn
    )


print()
print("DONE")
